"""Azure Resource Provisioner for Understudy GPU Training

Automatically provisions all required Azure resources when training is first requested.
"""

import os
import logging
from typing import Dict, Any, Optional
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.authorization import AuthorizationManagementClient
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, AzureError

logger = logging.getLogger(__name__)


class AzureResourceProvisioner:
    """Provisions and manages Azure resources for GPU training"""
    
    def __init__(self):
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP", "understudy-training")
        self.location = os.getenv("AZURE_LOCATION", "eastus")
        self.storage_account = os.getenv("AZURE_STORAGE_ACCOUNT")
        self.storage_container = os.getenv("AZURE_STORAGE_CONTAINER", "training-data")
        
        self.resources_provisioned = False
        self.credential = None
        
    def _init_clients(self):
        """Initialize Azure management clients"""
        if not self.subscription_id or not self.storage_account:
            raise ValueError(
                "Azure configuration incomplete. Required: "
                "AZURE_SUBSCRIPTION_ID and AZURE_STORAGE_ACCOUNT"
            )
        
        self.credential = DefaultAzureCredential()
        self.resource_client = ResourceManagementClient(
            self.credential, self.subscription_id
        )
        self.storage_client = StorageManagementClient(
            self.credential, self.subscription_id
        )
        self.network_client = NetworkManagementClient(
            self.credential, self.subscription_id
        )
        self.auth_client = AuthorizationManagementClient(
            self.credential, self.subscription_id
        )
    
    async def provision_resources(self) -> Dict[str, Any]:
        """Provision all required Azure resources"""
        if self.resources_provisioned:
            return {"status": "already_provisioned"}
        
        try:
            self._init_clients()
            
            logger.info("Starting Azure resource provisioning...")
            
            # 1. Create Resource Group
            rg_result = await self._create_resource_group()
            
            # 2. Create Storage Account
            storage_result = await self._create_storage_account()
            
            # 3. Create Storage Container
            container_result = await self._create_storage_container()
            
            # 4. Create Virtual Network
            vnet_result = await self._create_virtual_network()
            
            # 5. Create Network Security Group
            nsg_result = await self._create_network_security_group()
            
            # 6. Set up SSH key if not provided
            ssh_key = await self._setup_ssh_key()
            
            self.resources_provisioned = True
            
            result = {
                "status": "success",
                "resource_group": rg_result,
                "storage_account": storage_result,
                "storage_container": container_result,
                "virtual_network": vnet_result,
                "network_security_group": nsg_result,
                "ssh_key_configured": bool(ssh_key)
            }
            
            logger.info(f"Azure resources provisioned successfully: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to provision Azure resources: {e}")
            raise
    
    async def _create_resource_group(self) -> Dict[str, str]:
        """Create or update resource group"""
        try:
            rg_params = {
                "location": self.location,
                "tags": {
                    "project": "understudy",
                    "purpose": "gpu-training",
                    "managed_by": "understudy-backend"
                }
            }
            
            result = self.resource_client.resource_groups.create_or_update(
                self.resource_group,
                rg_params
            )
            
            logger.info(f"Resource group '{self.resource_group}' ready in {self.location}")
            return {
                "name": self.resource_group,
                "location": result.location,
                "status": "ready"
            }
            
        except Exception as e:
            logger.error(f"Failed to create resource group: {e}")
            raise
    
    async def _create_storage_account(self) -> Dict[str, str]:
        """Create storage account for training data"""
        try:
            # Check if storage account exists
            try:
                storage_account = self.storage_client.storage_accounts.get_properties(
                    self.resource_group,
                    self.storage_account
                )
                logger.info(f"Storage account '{self.storage_account}' already exists")
                return {
                    "name": self.storage_account,
                    "status": "existing",
                    "primary_endpoint": storage_account.primary_endpoints.blob
                }
            except:
                pass  # Storage account doesn't exist, create it
            
            # Create storage account
            storage_params = {
                "sku": {"name": "Standard_LRS"},  # Locally redundant storage
                "kind": "StorageV2",
                "location": self.location,
                "encryption": {
                    "services": {
                        "blob": {"enabled": True}
                    },
                    "key_source": "Microsoft.Storage"
                },
                "tags": {
                    "project": "understudy",
                    "purpose": "training-data"
                }
            }
            
            async_operation = self.storage_client.storage_accounts.begin_create(
                self.resource_group,
                self.storage_account,
                storage_params
            )
            
            storage_account = async_operation.result()
            
            logger.info(f"Storage account '{self.storage_account}' created")
            return {
                "name": self.storage_account,
                "status": "created",
                "primary_endpoint": storage_account.primary_endpoints.blob
            }
            
        except Exception as e:
            logger.error(f"Failed to create storage account: {e}")
            raise
    
    async def _create_storage_container(self) -> Dict[str, str]:
        """Create blob container for training data"""
        try:
            # Get storage account keys
            keys = self.storage_client.storage_accounts.list_keys(
                self.resource_group,
                self.storage_account
            )
            storage_key = keys.keys[0].value
            
            # Create blob service client
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={self.storage_account};"
                f"AccountKey={storage_key};"
                f"EndpointSuffix=core.windows.net"
            )
            
            blob_service_client = BlobServiceClient.from_connection_string(
                connection_string
            )
            
            # Create container
            try:
                container_client = blob_service_client.create_container(
                    self.storage_container
                )
                logger.info(f"Container '{self.storage_container}' created")
                status = "created"
            except ResourceExistsError:
                logger.info(f"Container '{self.storage_container}' already exists")
                status = "existing"
            
            # Store connection string for later use
            os.environ["AZURE_STORAGE_CONNECTION_STRING"] = connection_string
            
            return {
                "name": self.storage_container,
                "status": status
            }
            
        except Exception as e:
            logger.error(f"Failed to create storage container: {e}")
            raise
    
    async def _create_virtual_network(self) -> Dict[str, str]:
        """Create virtual network for VMs"""
        try:
            vnet_name = f"{self.resource_group}-vnet"
            subnet_name = "gpu-subnet"
            
            # Check if VNet exists
            try:
                vnet = self.network_client.virtual_networks.get(
                    self.resource_group,
                    vnet_name
                )
                logger.info(f"Virtual network '{vnet_name}' already exists")
                return {
                    "name": vnet_name,
                    "status": "existing",
                    "address_space": vnet.address_space.address_prefixes[0]
                }
            except:
                pass  # VNet doesn't exist, create it
            
            # Create VNet
            vnet_params = {
                "location": self.location,
                "address_space": {
                    "address_prefixes": ["10.0.0.0/16"]
                },
                "subnets": [{
                    "name": subnet_name,
                    "address_prefix": "10.0.1.0/24"
                }],
                "tags": {
                    "project": "understudy",
                    "purpose": "gpu-training"
                }
            }
            
            async_vnet = self.network_client.virtual_networks.begin_create_or_update(
                self.resource_group,
                vnet_name,
                vnet_params
            )
            
            vnet = async_vnet.result()
            
            logger.info(f"Virtual network '{vnet_name}' created")
            return {
                "name": vnet_name,
                "status": "created",
                "address_space": "10.0.0.0/16",
                "subnet": subnet_name
            }
            
        except Exception as e:
            logger.error(f"Failed to create virtual network: {e}")
            raise
    
    async def _create_network_security_group(self) -> Dict[str, str]:
        """Create network security group with SSH access"""
        try:
            nsg_name = f"{self.resource_group}-nsg"
            
            # Check if NSG exists
            try:
                nsg = self.network_client.network_security_groups.get(
                    self.resource_group,
                    nsg_name
                )
                logger.info(f"Network security group '{nsg_name}' already exists")
                return {
                    "name": nsg_name,
                    "status": "existing"
                }
            except:
                pass  # NSG doesn't exist, create it
            
            # Create NSG with SSH rule
            nsg_params = {
                "location": self.location,
                "security_rules": [{
                    "name": "SSH",
                    "priority": 1000,
                    "direction": "Inbound",
                    "access": "Allow",
                    "protocol": "Tcp",
                    "source_port_range": "*",
                    "destination_port_range": "22",
                    "source_address_prefix": "*",  # Restrict this in production
                    "destination_address_prefix": "*"
                }],
                "tags": {
                    "project": "understudy",
                    "purpose": "gpu-training"
                }
            }
            
            async_nsg = self.network_client.network_security_groups.begin_create_or_update(
                self.resource_group,
                nsg_name,
                nsg_params
            )
            
            nsg = async_nsg.result()
            
            logger.info(f"Network security group '{nsg_name}' created")
            return {
                "name": nsg_name,
                "status": "created"
            }
            
        except Exception as e:
            logger.error(f"Failed to create network security group: {e}")
            raise
    
    async def _setup_ssh_key(self) -> Optional[str]:
        """Generate or retrieve SSH key for VM access"""
        ssh_key = os.getenv("AZURE_SSH_PUBLIC_KEY")
        
        if not ssh_key:
            # Generate SSH key pair if not provided
            try:
                import subprocess
                
                ssh_dir = os.path.expanduser("~/.ssh")
                os.makedirs(ssh_dir, exist_ok=True)
                
                key_path = os.path.join(ssh_dir, "understudy_azure_key")
                
                if not os.path.exists(key_path):
                    # Generate new SSH key
                    subprocess.run([
                        "ssh-keygen",
                        "-t", "rsa",
                        "-b", "4096",
                        "-f", key_path,
                        "-N", "",  # No passphrase
                        "-C", "understudy@azure"
                    ], check=True)
                    
                    logger.info(f"Generated SSH key at {key_path}")
                
                # Read public key
                with open(f"{key_path}.pub", "r") as f:
                    ssh_key = f.read().strip()
                
                # Set environment variable for future use
                os.environ["AZURE_SSH_PUBLIC_KEY"] = ssh_key
                os.environ["AZURE_SSH_PRIVATE_KEY_PATH"] = key_path
                
                logger.info("SSH key configured successfully")
                
            except Exception as e:
                logger.warning(f"Could not generate SSH key: {e}")
                logger.warning("VM access will require manual SSH key configuration")
                return None
        
        return ssh_key
    
    async def validate_resources(self) -> Dict[str, bool]:
        """Validate that all required resources exist and are accessible"""
        try:
            self._init_clients()
            
            validations = {
                "resource_group": False,
                "storage_account": False,
                "storage_container": False,
                "virtual_network": False,
                "network_security_group": False,
                "ssh_key": False
            }
            
            # Check resource group
            try:
                self.resource_client.resource_groups.get(self.resource_group)
                validations["resource_group"] = True
            except:
                pass
            
            # Check storage account
            try:
                self.storage_client.storage_accounts.get_properties(
                    self.resource_group,
                    self.storage_account
                )
                validations["storage_account"] = True
            except:
                pass
            
            # Check SSH key
            validations["ssh_key"] = bool(
                os.getenv("AZURE_SSH_PUBLIC_KEY") or 
                os.path.exists(os.path.expanduser("~/.ssh/understudy_azure_key.pub"))
            )
            
            # Check virtual network
            try:
                vnet_name = f"{self.resource_group}-vnet"
                self.network_client.virtual_networks.get(
                    self.resource_group,
                    vnet_name
                )
                validations["virtual_network"] = True
            except:
                pass
            
            # Check NSG
            try:
                nsg_name = f"{self.resource_group}-nsg"
                self.network_client.network_security_groups.get(
                    self.resource_group,
                    nsg_name
                )
                validations["network_security_group"] = True
            except:
                pass
            
            # Check container (if storage account exists)
            if validations["storage_account"]:
                try:
                    keys = self.storage_client.storage_accounts.list_keys(
                        self.resource_group,
                        self.storage_account
                    )
                    storage_key = keys.keys[0].value
                    
                    blob_service_client = BlobServiceClient(
                        account_url=f"https://{self.storage_account}.blob.core.windows.net",
                        credential=storage_key
                    )
                    
                    container_client = blob_service_client.get_container_client(
                        self.storage_container
                    )
                    container_client.get_container_properties()
                    validations["storage_container"] = True
                except:
                    pass
            
            all_valid = all(validations.values())
            self.resources_provisioned = all_valid
            
            return {
                "all_resources_valid": all_valid,
                "validations": validations
            }
            
        except Exception as e:
            logger.error(f"Resource validation failed: {e}")
            return {
                "all_resources_valid": False,
                "error": str(e)
            }
    
    async def cleanup_resources(self, delete_resource_group: bool = False):
        """Clean up Azure resources"""
        try:
            self._init_clients()
            
            if delete_resource_group:
                # Delete entire resource group (and all resources within)
                async_delete = self.resource_client.resource_groups.begin_delete(
                    self.resource_group
                )
                async_delete.result()
                logger.info(f"Deleted resource group '{self.resource_group}' and all resources")
            else:
                # Just clean up VMs and their resources
                vms = self.resource_client.resources.list_by_resource_group(
                    self.resource_group,
                    filter="resourceType eq 'Microsoft.Compute/virtualMachines'"
                )
                
                for vm in vms:
                    try:
                        vm_name = vm.name
                        # Delete VM
                        self.resource_client.resources.begin_delete_by_id(
                            vm.id,
                            api_version="2021-03-01"
                        ).result()
                        logger.info(f"Deleted VM: {vm_name}")
                    except:
                        pass
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Global provisioner instance
azure_provisioner = AzureResourceProvisioner()