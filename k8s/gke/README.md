# Deploying to GKE


## Requires

- gcloud auth login
- GCP project with billing
- kubectl installed

## Set projects + APIs

gcloud config set project PROJECT_ID
gcloud services enable container.googleapis.com artifactregistry.googleapis.com

## Creating Artifact Registry repository

gcloud artifacts repositories create amf-rag-agent --repository-format=docker --location=REGION

## Building and pushing image

gcloud auth configure-docker REGION-docker.pkg.dev
docker build -f Dockerfile.cpu -t REGION-docker.pkg.dev/PROJECT_ID/amf-rag/amf-rag-api:cpu .
docker push REGION-docker.pkg.dev/PROJECT_ID/amf-rag/amf-rag-api:cpu

## Creating cluster

gcloud container clusters create amf-rag-cluster --num-nodes=2 --machine-type=e2-standard-2 --region=REGION

## Creating secrets

kubectl create secret generic amf-rag-secrets --from-literal=ANTHROPIC_API_KEY=... --from-literal=OPENAI_API_KEY= ... \
    --from-literal=API_KEYS=dev-key-12345 -from-literal=ELASTICSEARCH_URL=... --from-literal=REDIS_URL ...

## Deploying

kubectl apply -f k8s/gke/deployment.yaml
kubectl apply -f k8s/gke/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl get service amf-rag-api # external IP

## Deploying with GPU

Installing Nvidia DaemonSet from this repo and follow instructions from the README.md : https://github.com/nvidia/k8s-device-plugin

gcloud container node-pools create gpu-pool --cluster=amf-rag-cluster --machine-type=n1-standard-4 --accelerator=type=nvidia-tesla-t4,count=1 \
    --num-nodes=1 --region=REGION
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
for the deployment GPU yaml adds:
    ressources: { limits: { nvidia.com/gpu: 1 } }
    nodeSelector: { cloud.google.com/gke-accelerator: nvidia-tesla-t4 }
and image : {amf-rag-api:gpu}