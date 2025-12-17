# Ray Infrastructure Setup

1. Have the GPU nodes and GPU operator

2. Install Kueue and Ray
From Operator Hub, in operator Red Hat OpenShift AI, go to Data Science cluster and create Data Science cluster.


2. Ray + Kueue CRDs
Check that required CRD exist
$oc get crd | grep ray
# rayclusters.ray.io
# rayjobs.ray.io
# rayservices.ray.io
# rays.components.platform.opendatahub.io

$ oc get crd | grep kueue
# clusterqueues.kueue.x-k8s.io
# localqueues.kueue.x-k8s.io
# resourceflavors.kueue.x-k8s.io

3. Ray Cluster may need to be connected with S3 Buckets

#Pre-requisites
1. Create S3 Bucket and IAM User with policy

#Create Secret in OpenShift Cluster
2. Create OC secret In OpenShift Ray namespace
    $ $oc create secret generic aws-creds \
    2.   -n ray-finetune-llm-deepspeed002 \
    3.   --from-literal=AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID \
    4.   --from-literal=AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY \
    5.   --from-literal=AWS_REGION=us-east-1
#Update Cluster YAML to include secret info

3. Update 02_ray_localqueue_and_cluster.yaml with secret info
$ Refer secretRef with name: aws-creds (this should be name of your secret in your OpenShift cluster)

4. Set Cluster
    $oc apply -f 02_ray_localqueue_and_cluster.yaml
    $oc $delete pod -n ray-finetune-llm-deepspeed002 -l ray.io/cluster=ray
    # wait for head+workers to come back



3. Per Namespace Setup

NS="ray-finetune-llm-deepspeed002"
oc new-project "$NS"


3. Run Instructor Commands at path: 
$ cluster-prereqs/Instructor_Commands

Create a Ray cluster in OpenShift AI Console


4. Download Llama 3.2 1B model on local machine and upload to OpenShift

# Get your HF token
update the token in the download_llama.py file

#Download Llama Model
$python cluster-prereqs/scripts/download_llama.py

#Upload Llama Model to OpenShift
$ oc cp ./llama-3.2-1b-instruct notebook-0:/opt/app-root/src/models/llama-3.2-1b-instruct -n ray-finetune-llm-deepspeed002