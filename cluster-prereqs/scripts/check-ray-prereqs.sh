#!/usr/bin/env bash
NS="${1:-ray-finetune-llm-deepspeed002}"

echo "=== GPU nodes & taints ==="
oc get nodes -l node.kubernetes.io/instance-type=g6.2xlarge \
  -o custom-columns=NAME:.metadata.name,TAINTS:.spec.taints

echo -e "\n=== ResourceFlavor & ClusterQueue ==="
oc get resourceflavors.kueue.x-k8s.io -n "$NS" || echo "(no flavors)"
oc get clusterqueues.kueue.x-k8s.io || echo "(no clusterqueues)"

echo -e "\n=== LocalQueue & RayCluster ==="
oc get localqueues.kueue.x-k8s.io -n "$NS" || echo "(no localqueues)"
oc get rayclusters -n "$NS" -o wide || echo "(no rayclusters)"

echo -e "\n=== Ray pods ==="
oc get pods -n "$NS" | grep ray || echo "no ray pods"

HEAD_POD=$(oc get pods -n "$NS" -o name 2>/dev/null | grep ray-head || true)
if [ -n "$HEAD_POD" ]; then
  echo -e "\n=== Head pod events (last 20 lines) ==="
  oc describe "$HEAD_POD" -n "$NS" | tail -n 20
else
  echo -e "\n(no head pod currently)\n"
fi
