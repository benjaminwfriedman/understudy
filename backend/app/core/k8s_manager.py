"""
Kubernetes API Manager

Manages SLM inference services through the Kubernetes API.
Creates and manages Deployments and Jobs for model inference.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)


class K8sManager:
    """Manages Kubernetes resources for SLM inference services."""
    
    def __init__(self):
        self.namespace = os.getenv("NAMESPACE", "understudy")
        self.model_broker_url = os.getenv("MODEL_BROKER_SERVICE_URL", "http://model-broker-service:8003")
        
        # Initialize Kubernetes client
        try:
            if os.getenv("K8S_IN_CLUSTER", "false").lower() == "true":
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes configuration")
            else:
                config.load_kube_config()
                logger.info("Loaded local Kubernetes configuration")
                
            self.apps_v1 = client.AppsV1Api()
            self.batch_v1 = client.BatchV1Api()
            self.core_v1 = client.CoreV1Api()
            self.autoscaling_v2 = client.AutoscalingV2Api()
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
    
    def create_slm_deployment(
        self,
        endpoint_id: str,
        version: int,
        model_path: str,
        deployment_threshold: float = 0.95
    ) -> Dict[str, Any]:
        """Create a persistent SLM deployment for serving requests."""
        deployment_name = f"slm-{endpoint_id}-v{version}"
        service_name = f"slm-{endpoint_id}-svc"
        hpa_name = f"slm-{endpoint_id}-hpa"
        
        try:
            # Create deployment
            deployment = self._build_deployment_spec(
                name=deployment_name,
                endpoint_id=endpoint_id,
                version=version,
                model_path=model_path,
                mode="endpoint"
            )
            
            self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )
            logger.info(f"Created deployment: {deployment_name}")
            
            # Create service
            service = self._build_service_spec(
                name=service_name,
                endpoint_id=endpoint_id,
                selector_labels={
                    "app": "slm-inference",
                    "endpoint_id": endpoint_id
                }
            )
            
            self.core_v1.create_namespaced_service(
                namespace=self.namespace,
                body=service
            )
            logger.info(f"Created service: {service_name}")
            
            # Create HorizontalPodAutoscaler
            hpa = self._build_hpa_spec(
                name=hpa_name,
                deployment_name=deployment_name
            )
            
            self.autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                namespace=self.namespace,
                body=hpa
            )
            logger.info(f"Created HPA: {hpa_name}")
            
            return {
                "deployment_name": deployment_name,
                "service_name": service_name,
                "hpa_name": hpa_name,
                "status": "created"
            }
            
        except ApiException as e:
            logger.error(f"Failed to create SLM deployment: {e}")
            raise
    
    def create_slm_batch_job(
        self,
        train_id: str,
        endpoint_id: str,
        version: int,
        model_path: str,
        evaluation_batch_id: str
    ) -> Dict[str, Any]:
        """Create a batch job for model evaluation."""
        # Include timestamp to make job name unique
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        job_name = f"slm-eval-{train_id[:8]}-{timestamp}"
        
        try:
            # Check if a job with similar name exists and delete it if completed
            try:
                existing_jobs = self.batch_v1.list_namespaced_job(
                    namespace=self.namespace,
                    label_selector=f"train_id={train_id},app=slm-inference,mode=batch"
                )
                for existing_job in existing_jobs.items:
                    if existing_job.status.succeeded or existing_job.status.failed:
                        # Delete completed/failed jobs
                        logger.info(f"Deleting old job: {existing_job.metadata.name}")
                        self.batch_v1.delete_namespaced_job(
                            name=existing_job.metadata.name,
                            namespace=self.namespace,
                            propagation_policy="Background"
                        )
            except ApiException as e:
                logger.debug(f"No existing jobs to clean up: {e}")
            
            job = self._build_job_spec(
                name=job_name,
                train_id=train_id,
                endpoint_id=endpoint_id,
                version=version,
                model_path=model_path,
                evaluation_batch_id=evaluation_batch_id
            )
            
            self.batch_v1.create_namespaced_job(
                namespace=self.namespace,
                body=job
            )
            logger.info(f"Created batch job: {job_name}")
            
            return {
                "job_name": job_name,
                "status": "created",
                "train_id": train_id
            }
            
        except ApiException as e:
            logger.error(f"Failed to create batch job: {e}")
            raise
    
    def delete_slm_deployment(self, deployment_name: str) -> bool:
        """Delete an SLM deployment and associated resources."""
        try:
            # Delete deployment
            self.apps_v1.delete_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            logger.info(f"Deleted deployment: {deployment_name}")
            
            # Delete associated service
            service_name = deployment_name.replace("slm-", "slm-").replace("-v", "-svc")
            try:
                self.core_v1.delete_namespaced_service(
                    name=service_name,
                    namespace=self.namespace
                )
                logger.info(f"Deleted service: {service_name}")
            except ApiException:
                pass  # Service might not exist
            
            # Delete HPA
            hpa_name = deployment_name.replace("slm-", "slm-").replace("-v", "-hpa")
            try:
                self.autoscaling_v2.delete_namespaced_horizontal_pod_autoscaler(
                    name=hpa_name,
                    namespace=self.namespace
                )
                logger.info(f"Deleted HPA: {hpa_name}")
            except ApiException:
                pass  # HPA might not exist
            
            return True
            
        except ApiException as e:
            logger.error(f"Failed to delete deployment {deployment_name}: {e}")
            return False
    
    def get_deployment_status(self, deployment_name: str) -> Optional[Dict[str, Any]]:
        """Get the status of a deployment."""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            return {
                "name": deployment.metadata.name,
                "ready_replicas": deployment.status.ready_replicas or 0,
                "replicas": deployment.status.replicas or 0,
                "available_replicas": deployment.status.available_replicas or 0,
                "conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason
                    }
                    for condition in (deployment.status.conditions or [])
                ]
            }
            
        except ApiException as e:
            if e.status == 404:
                return None
            logger.error(f"Failed to get deployment status: {e}")
            raise
    
    def get_job_status(self, job_name: str) -> Optional[Dict[str, Any]]:
        """Get the status of a batch job."""
        try:
            job = self.batch_v1.read_namespaced_job(
                name=job_name,
                namespace=self.namespace
            )
            
            return {
                "name": job.metadata.name,
                "active": job.status.active or 0,
                "succeeded": job.status.succeeded or 0,
                "failed": job.status.failed or 0,
                "completion_time": job.status.completion_time,
                "start_time": job.status.start_time,
                "conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason
                    }
                    for condition in (job.status.conditions or [])
                ]
            }
            
        except ApiException as e:
            if e.status == 404:
                return None
            logger.error(f"Failed to get job status: {e}")
            raise
    
    def cleanup_completed_jobs(self) -> int:
        """Clean up completed batch jobs to prevent namespace clutter."""
        try:
            jobs = self.batch_v1.list_namespaced_job(
                namespace=self.namespace,
                label_selector="app=slm-inference,mode=batch"
            )
            
            cleaned_count = 0
            for job in jobs.items:
                # Delete if succeeded or failed
                if job.status.succeeded or job.status.failed:
                    try:
                        self.batch_v1.delete_namespaced_job(
                            name=job.metadata.name,
                            namespace=self.namespace,
                            propagation_policy="Background"
                        )
                        logger.info(f"Cleaned up completed job: {job.metadata.name}")
                        cleaned_count += 1
                    except ApiException as e:
                        logger.warning(f"Failed to cleanup job {job.metadata.name}: {e}")
            
            return cleaned_count
            
        except ApiException as e:
            logger.error(f"Failed to cleanup jobs: {e}")
            return 0
    
    def list_slm_deployments(self) -> List[Dict[str, Any]]:
        """List all SLM deployments."""
        try:
            deployments = self.apps_v1.list_namespaced_deployment(
                namespace=self.namespace,
                label_selector="app=slm-inference,mode=endpoint"
            )
            
            result = []
            for deployment in deployments.items:
                labels = deployment.metadata.labels or {}
                result.append({
                    "name": deployment.metadata.name,
                    "endpoint_id": labels.get("endpoint_id"),
                    "version": labels.get("version"),
                    "ready_replicas": deployment.status.ready_replicas or 0,
                    "replicas": deployment.status.replicas or 0,
                    "created": deployment.metadata.creation_timestamp
                })
            
            return result
            
        except ApiException as e:
            logger.error(f"Failed to list SLM deployments: {e}")
            return []
    
    def _build_deployment_spec(
        self,
        name: str,
        endpoint_id: str,
        version: int,
        model_path: str,
        mode: str = "endpoint"
    ) -> client.V1Deployment:
        """
        Build deployment specification.

        This defines the k8s spec for the slm deployment
        
        """
        labels = {
            "app": "slm-inference",
            "endpoint_id": endpoint_id,
            "version": str(version),
            "mode": mode
        }
        
        container = client.V1Container(
            name="slm-server",
            image="vllm/vllm-openai:latest",
            command=["python", "-m", "vllm.entrypoints.openai.api_server"],
            args=[
                f"--model={model_path}",
                "--port=8000",
                "--host=0.0.0.0"
            ],
            ports=[client.V1ContainerPort(container_port=8000)],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "4", "memory": "8Gi"},
                limits={"cpu": "8", "memory": "16Gi"}
            ),
            env=[
                client.V1EnvVar(name="MODEL_BROKER_URL", value=self.model_broker_url),
                client.V1EnvVar(name="ENDPOINT_ID", value=endpoint_id),
                client.V1EnvVar(name="VERSION", value=str(version))
            ],
            volume_mounts=[
                client.V1VolumeMount(
                    name="model-store",
                    mount_path="/models",
                    read_only=True
                )
            ]
        )
        
        pod_spec = client.V1PodSpec(
            containers=[container],
            volumes=[
                client.V1Volume(
                    name="model-store",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name="model-weights-pvc"
                    )
                )
            ]
        )
        
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels),
            spec=pod_spec
        )
        
        spec = client.V1DeploymentSpec(
            replicas=1,
            selector=client.V1LabelSelector(match_labels={
                "app": "slm-inference",
                "endpoint_id": endpoint_id
            }),
            template=template
        )
        
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name=name, labels=labels),
            spec=spec
        )
        
        return deployment
    
    def _build_service_spec(
        self,
        name: str,
        endpoint_id: str,
        selector_labels: Dict[str, str]
    ) -> client.V1Service:
        """Build service specification."""
        spec = client.V1ServiceSpec(
            selector=selector_labels,
            ports=[
                client.V1ServicePort(
                    name="http",
                    port=80,
                    target_port=8000
                )
            ],
            type="ClusterIP"
        )
        
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(name=name),
            spec=spec
        )
        
        return service
    
    def _build_hpa_spec(
        self,
        name: str,
        deployment_name: str,
        min_replicas: int = 1,
        max_replicas: int = 5
    ) -> client.V2HorizontalPodAutoscaler:
        """Build HorizontalPodAutoscaler specification."""
        spec = client.V2HorizontalPodAutoscalerSpec(
            scale_target_ref=client.V2CrossVersionObjectReference(
                api_version="apps/v1",
                kind="Deployment",
                name=deployment_name
            ),
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            metrics=[
                client.V2MetricSpec(
                    type="Resource",
                    resource=client.V2ResourceMetricSource(
                        name="cpu",
                        target=client.V2MetricTarget(
                            type="Utilization",
                            average_utilization=70
                        )
                    )
                )
            ]
        )
        
        hpa = client.V2HorizontalPodAutoscaler(
            api_version="autoscaling/v2",
            kind="HorizontalPodAutoscaler",
            metadata=client.V1ObjectMeta(name=name),
            spec=spec
        )
        
        return hpa
    
    def _build_job_spec(
        self,
        name: str,
        train_id: str,
        endpoint_id: str,
        version: int,
        model_path: str,
        evaluation_batch_id: str
    ) -> client.V1Job:
        """Build batch job specification."""
        labels = {
            "app": "slm-inference",
            "mode": "batch",
            "train_id": train_id,
            "endpoint_id": endpoint_id
        }
        
        container = client.V1Container(
            name="slm-inference",
            image="bennyfriedman/understudy-slm-inference:latest-arm64",
            image_pull_policy="Always",
            env=[
                client.V1EnvVar(name="MODEL_PATH", value="/models"),
                client.V1EnvVar(name="ENDPOINT_ID", value=endpoint_id),
                client.V1EnvVar(name="VERSION", value=str(version)),
                client.V1EnvVar(name="MODE", value="batch"),
                client.V1EnvVar(name="EVAL_BATCH_ID", value=evaluation_batch_id),
                client.V1EnvVar(name="TRAIN_ID", value=train_id),
                client.V1EnvVar(name="BACKEND_URL", value=os.getenv("BACKEND_URL", "http://backend-service:8000")),
                client.V1EnvVar(name="MODEL_BROKER_URL", value=self.model_broker_url),
                client.V1EnvVar(name="HF_TOKEN", value=os.getenv("HF_TOKEN", ""))
            ],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "1", "memory": "6Gi"},
                limits={"cpu": "1", "memory": "6Gi"}
            )
        )
        
        pod_spec = client.V1PodSpec(
            restart_policy="Never",
            containers=[container]
        )
        
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels),
            spec=pod_spec
        )
        
        spec = client.V1JobSpec(
            template=template,
            backoff_limit=3,
            ttl_seconds_after_finished=3600  # 1 hour - reduced from 24 hours for faster cleanup
        )
        
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=name, labels=labels),
            spec=spec
        )
        
        return job


# Global instance
k8s_manager = None

def get_k8s_manager() -> K8sManager:
    """Get the global K8s manager instance."""
    global k8s_manager
    if k8s_manager is None:
        k8s_manager = K8sManager()
    return k8s_manager