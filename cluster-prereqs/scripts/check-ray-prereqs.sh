#!/usr/bin/env bash
# Enhanced Ray / GPU environment checker for the Llama3 lab
# Includes DataScienceCluster validation for Ray and Kueue

set -uo pipefail

NS="${1:-ray-finetune-llm-deepspeed002}"

echo "🔎 Checking Ray prerequisites in namespace: $NS"
echo "=================================================="
echo ""

ok()  { echo "  ✅ $*"; }
warn(){ echo "  ⚠️  $*"; }
bad() { echo "  ❌ $*"; }
info(){ echo "  ℹ️  $*"; }

# Track overall status
CRITICAL_ERRORS=0
WARNINGS=0

# ============================================================================
# NEW SECTION: OpenShift AI Platform Check (DataScienceCluster)
# ============================================================================
echo "0) OpenShift AI Platform Configuration"
echo "--------------------------------------"

# Check if DataScienceCluster CRD exists
if ! oc get crd datascienceclusters.datasciencecluster.opendatahub.io >/dev/null 2>&1; then
  bad "DataScienceCluster CRD not found - OpenShift AI may not be installed!"
  warn "→ Install Red Hat OpenShift AI from OperatorHub first."
  CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
else
  ok "DataScienceCluster CRD exists (OpenShift AI is installed)."
  
  # Check if default DSC exists
  if ! oc get datasciencecluster default-dsc >/dev/null 2>&1; then
    bad "DataScienceCluster 'default-dsc' not found!"
    warn "→ OpenShift AI operator may still be initializing, or installation failed."
    CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
  else
    ok "DataScienceCluster 'default-dsc' exists."
    
    # Check if jq is available for JSON parsing
    if ! command -v jq >/dev/null 2>&1; then
      warn "jq not installed - using fallback JSON parsing."
      info "Install jq for better output: sudo yum install jq -y"
      
      # Fallback: use oc jsonpath
      RAY_STATE=$(oc get datasciencecluster default-dsc -o jsonpath='{.spec.components.ray.managementState}' 2>/dev/null)
      KUEUE_STATE=$(oc get datasciencecluster default-dsc -o jsonpath='{.spec.components.kueue.managementState}' 2>/dev/null)
      
    else
      # Use jq for cleaner output
      COMPONENTS=$(oc get datasciencecluster default-dsc -o json 2>/dev/null | \
        jq -r '.spec.components | {ray: .ray.managementState, kueue: .kueue.managementState}' 2>/dev/null)
      
      if [ $? -eq 0 ] && [ -n "$COMPONENTS" ]; then
        echo ""
        info "Component States:"
        echo "$COMPONENTS" | sed 's/^/    /'
        echo ""
      fi
      
      RAY_STATE=$(oc get datasciencecluster default-dsc -o jsonpath='{.spec.components.ray.managementState}' 2>/dev/null)
      KUEUE_STATE=$(oc get datasciencecluster default-dsc -o jsonpath='{.spec.components.kueue.managementState}' 2>/dev/null)
    fi
    
    # Validate Ray component
    echo "  Ray Component:"
    if [ "$RAY_STATE" = "Managed" ]; then
      ok "Ray is ENABLED (managementState: Managed)"
      
      # Check if Ray operator is actually deployed
      if oc get deployment kuberay-operator -n redhat-ods-applications >/dev/null 2>&1; then
        READY=$(oc get deployment kuberay-operator -n redhat-ods-applications -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
        DESIRED=$(oc get deployment kuberay-operator -n redhat-ods-applications -o jsonpath='{.status.replicas}' 2>/dev/null)
        
        if [ "$READY" = "$DESIRED" ] && [ "$READY" != "0" ]; then
          ok "Ray Operator is deployed and ready ($READY/$DESIRED replicas)"
        else
          warn "Ray Operator exists but not ready yet ($READY/$DESIRED replicas)"
          info "Wait 2-5 minutes for operator to become ready."
          WARNINGS=$((WARNINGS + 1))
        fi
      else
        bad "Ray is enabled in DSC but operator deployment not found!"
        warn "→ Check: oc get pods -n redhat-ods-applications | grep kuberay"
        CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
      fi
      
    elif [ "$RAY_STATE" = "Removed" ]; then
      bad "Ray is DISABLED (managementState: Removed)"
      warn "→ Enable Ray with: oc patch datasciencecluster default-dsc --type=merge -p '{\"spec\":{\"components\":{\"ray\":{\"managementState\":\"Managed\"}}}}'"
      CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
      
    else
      bad "Ray managementState is unexpected: '$RAY_STATE'"
      warn "→ Expected 'Managed' or 'Removed', got '$RAY_STATE'"
      CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
    fi
    
    # Validate Kueue component
    echo ""
    echo "  Kueue Component:"
    if [ "$KUEUE_STATE" = "Managed" ]; then
      ok "Kueue is ENABLED (managementState: Managed)"
      
      # Check if Kueue operator is actually deployed
      if oc get deployment kueue-controller-manager -n redhat-ods-applications >/dev/null 2>&1; then
        READY=$(oc get deployment kueue-controller-manager -n redhat-ods-applications -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
        DESIRED=$(oc get deployment kueue-controller-manager -n redhat-ods-applications -o jsonpath='{.status.replicas}' 2>/dev/null)
        
        if [ "$READY" = "$DESIRED" ] && [ "$READY" != "0" ]; then
          ok "Kueue Operator is deployed and ready ($READY/$DESIRED replicas)"
        else
          warn "Kueue Operator exists but not ready yet ($READY/$DESIRED replicas)"
          info "Wait 2-5 minutes for operator to become ready."
          WARNINGS=$((WARNINGS + 1))
        fi
      else
        bad "Kueue is enabled in DSC but operator deployment not found!"
        warn "→ Check: oc get pods -n redhat-ods-applications | grep kueue"
        CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
      fi
      
    elif [ "$KUEUE_STATE" = "Removed" ]; then
      bad "Kueue is DISABLED (managementState: Removed)"
      warn "→ Enable Kueue with: oc patch datasciencecluster default-dsc --type=merge -p '{\"spec\":{\"components\":{\"kueue\":{\"managementState\":\"Managed\"}}}}'"
      CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
      
    else
      bad "Kueue managementState is unexpected: '$KUEUE_STATE'"
      warn "→ Expected 'Managed' or 'Removed', got '$KUEUE_STATE'"
      CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
    fi
    
    # Check CRDs are installed
    echo ""
    echo "  Custom Resource Definitions:"
    
    if oc get crd rayclusters.ray.io >/dev/null 2>&1; then
      ok "RayCluster CRD is installed"
    else
      bad "RayCluster CRD not found - Ray operator may not be initialized yet"
      CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
    fi
    
    if oc get crd clusterqueues.kueue.x-k8s.io >/dev/null 2>&1; then
      ok "ClusterQueue CRD is installed"
    else
      bad "ClusterQueue CRD not found - Kueue operator may not be initialized yet"
      CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
    fi
    
    if oc get crd localqueues.kueue.x-k8s.io >/dev/null 2>&1; then
      ok "LocalQueue CRD is installed"
    else
      bad "LocalQueue CRD not found - Kueue operator may not be initialized yet"
      CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
    fi
    
    if oc get crd resourceflavors.kueue.x-k8s.io >/dev/null 2>&1; then
      ok "ResourceFlavor CRD is installed"
    else
      bad "ResourceFlavor CRD not found - Kueue operator may not be initialized yet"
      CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
    fi
  fi
fi

# If critical errors in platform setup, show summary and exit early
if [ $CRITICAL_ERRORS -gt 0 ]; then
  echo ""
  echo "⚠️  CRITICAL PLATFORM ISSUES DETECTED ⚠️"
  echo "----------------------------------------"
  echo "Found $CRITICAL_ERRORS critical error(s) in OpenShift AI platform setup."
  echo ""
  echo "🛠️  ACTION REQUIRED:"
  echo "  1. Ensure OpenShift AI is installed (OperatorHub → Red Hat OpenShift AI)"
  echo "  2. Enable Ray: oc patch datasciencecluster default-dsc --type=merge -p '{\"spec\":{\"components\":{\"ray\":{\"managementState\":\"Managed\"}}}}'"
  echo "  3. Enable Kueue: oc patch datasciencecluster default-dsc --type=merge -p '{\"spec\":{\"components\":{\"kueue\":{\"managementState\":\"Managed\"}}}}'"
  echo "  4. Wait 5 minutes for operators to deploy"
  echo "  5. Re-run this script"
  echo ""
  exit 1
fi

# ============================================================================
# ORIGINAL SECTIONS (with original numbering shifted)
# ============================================================================

echo ""
echo "1) GPU nodes & taints"
echo "---------------------"

gpu_nodes=$(oc get nodes -l node.kubernetes.io/instance-type=g6.2xlarge \
  --no-headers 2>/dev/null | wc -l | tr -d ' ')
if [ "$gpu_nodes" -eq 0 ]; then
  bad "No nodes with label node.kubernetes.io/instance-type=g6.2xlarge"
  warn "You either have no GPU nodes, or they are labeled differently."
  info "Check your GPU nodes: oc get nodes -l nvidia.com/gpu.present=true"
  WARNINGS=$((WARNINGS + 1))
else
  ok "$gpu_nodes GPU node(s) with instance-type=g6.2xlarge"
  oc get nodes -l node.kubernetes.io/instance-type=g6.2xlarge \
    -o custom-columns=NAME:.metadata.name,TAINTS:.spec.taints

  no_gpu_taint=$(oc get nodes -l node.kubernetes.io/instance-type=g6.2xlarge \
    -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.spec.taints}{"\n"}{end}' |
    grep -vc 'nvidia.com/gpu' || echo "0")
  if [ "$no_gpu_taint" -gt 0 ]; then
    warn "$no_gpu_taint GPU node(s) do NOT have nvidia.com/gpu taint – scheduling may be uneven."
    WARNINGS=$((WARNINGS + 1))
  else
    ok "All GPU nodes are tainted with nvidia.com/gpu (expected for workshop)."
  fi
fi

echo ""
echo "2) Kueue objects (ResourceFlavor + ClusterQueue + LocalQueue)"
echo "-------------------------------------------------------------"

# ResourceFlavor
if oc get resourceflavors.kueue.x-k8s.io default-flavor -n "$NS" >/dev/null 2>&1; then
  ok "ResourceFlavor 'default-flavor' exists in $NS."
  oc get resourceflavors.kueue.x-k8s.io default-flavor -n "$NS" -o yaml | \
    sed -n '1,40p'
else
  bad "ResourceFlavor 'default-flavor' NOT found in namespace $NS."
  warn "→ Apply: oc apply -f 01_gpu_flavor_and_queue.yaml"
  CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
fi

# ClusterQueue (cluster-scoped, no namespace)
if oc get clusterqueues.kueue.x-k8s.io ray-gpu-queue >/dev/null 2>&1; then
  ok "ClusterQueue 'ray-gpu-queue' exists."
else
  bad "ClusterQueue 'ray-gpu-queue' NOT found."
  warn "→ Apply: oc apply -f 01_gpu_flavor_and_queue.yaml"
  CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
fi

# LocalQueue
if oc get localqueues.kueue.x-k8s.io local-queue-ray -n "$NS" >/dev/null 2>&1; then
  cq=$(oc get localqueues.kueue.x-k8s.io local-queue-ray -n "$NS" -o jsonpath='{.spec.clusterQueue}')
  if [ "$cq" = "ray-gpu-queue" ]; then
    ok "LocalQueue 'local-queue-ray' points to ClusterQueue 'ray-gpu-queue'."
  else
    bad "LocalQueue 'local-queue-ray' points to '$cq' (expected 'ray-gpu-queue')."
    WARNINGS=$((WARNINGS + 1))
  fi
else
  bad "LocalQueue 'local-queue-ray' NOT found in $NS."
  warn "→ Apply: oc apply -f 02_ray_localqueue_and_cluster.yaml"
  CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
fi

echo ""
echo "3) RayCluster spec (tolerations, nodeSelector, queue label)"
echo "-----------------------------------------------------------"

if ! oc get rayclusters.ray.io -n "$NS" >/dev/null 2>&1; then
  bad "No RayCluster found in $NS. Did you apply 02_ray_localqueue_and_cluster.yaml?"
  CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
else
  oc get rayclusters.ray.io -n "$NS"

  if oc get rayclusters.ray.io ray -n "$NS" >/dev/null 2>&1; then
    rc_yaml=$(oc get rayclusters.ray.io ray -n "$NS" -o yaml)

    echo ""
    echo "  RayCluster 'ray' key fields:"
    echo "$rc_yaml" | sed -n '1,80p'

    # queue label
    if echo "$rc_yaml" | grep -q 'kueue.x-k8s.io/queue-name: local-queue-ray'; then
      ok "RayCluster has label kueue.x-k8s.io/queue-name=local-queue-ray."
    else
      bad "RayCluster is missing label kueue.x-k8s.io/queue-name=local-queue-ray."
      WARNINGS=$((WARNINGS + 1))
    fi

    # head tolerations
    if echo "$rc_yaml" | grep -A4 'headGroupSpec' | grep -q 'nvidia.com/gpu'; then
      ok "Head group has GPU toleration configured."
    else
      bad "Head group is missing GPU toleration (nvidia.com/gpu=True:NoSchedule)."
      WARNINGS=$((WARNINGS + 1))
    fi

    # worker tolerations
    if echo "$rc_yaml" | grep -A10 'workerGroupSpecs' | grep -q 'nvidia.com/gpu'; then
      ok "Worker group has GPU toleration configured."
    else
      bad "Worker group is missing GPU toleration (nvidia.com/gpu=True:NoSchedule)."
      WARNINGS=$((WARNINGS + 1))
    fi
  else
    bad "RayCluster named 'ray' not found in $NS."
    CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
  fi
fi

echo ""
echo "4) RBAC for notebook service account"
echo "------------------------------------"

if oc get sa notebook -n "$NS" >/dev/null 2>&1; then
  ok "ServiceAccount 'notebook' exists in $NS."
else
  bad "ServiceAccount 'notebook' NOT found in $NS."
  warn "→ Create workbench in OpenShift AI Dashboard, or create SA manually"
  WARNINGS=$((WARNINGS + 1))
fi

if oc get role raycluster-user -n "$NS" >/dev/null 2>&1; then
  ok "Role 'raycluster-user' exists."
else
  bad "Role 'raycluster-user' NOT found (03_rbac_notebook_ray.yaml missing?)."
  warn "→ Apply: oc apply -f 03_rbac_notebook_ray.yaml"
  CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
fi

if oc get rolebinding raycluster-user-binding -n "$NS" >/dev/null 2>&1; then
  ok "RoleBinding 'raycluster-user-binding' exists."
else
  bad "RoleBinding 'raycluster-user-binding' NOT found."
  warn "→ Apply: oc apply -f 03_rbac_notebook_ray.yaml"
  CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
fi

echo ""
echo "5) Ray pods & scheduler diagnosis"
echo "---------------------------------"

pods=$(oc get pods -n "$NS" --no-headers 2>/dev/null | grep ray || true)
if [ -z "$pods" ]; then
  warn "No Ray pods currently in namespace $NS (cluster might be suspended/not started)."
  info "This is normal if you haven't run the notebook yet."
else
  echo "$pods"
fi

HEAD_POD=$(oc get pods -n "$NS" -o name 2>/dev/null | grep ray-head || true)
if [ -n "$HEAD_POD" ]; then
  echo ""
  echo "  Head pod details ($HEAD_POD):"
  oc get "$HEAD_POD" -n "$NS" -o wide

  echo ""
  echo "  Last scheduling events:"
  ev=$(oc describe "$HEAD_POD" -n "$NS" | grep -A3 "FailedScheduling" || true)
  if [ -z "$ev" ]; then
    ok "No recent FailedScheduling events for head pod."
  else
    echo "$ev"
    if echo "$ev" | grep -q 'untolerated taint {nvidia.com/gpu'; then
      bad "Scheduler says GPU taint nvidia.com/gpu=True is NOT tolerated by the pod."
      warn "→ Check tolerations in 02_ray_localqueue_and_cluster.yaml and re-apply."
      CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
    fi
    if echo "$ev" | grep -q 'Insufficient nvidia.com/gpu'; then
      bad "Scheduler says there are not enough allocatable GPUs."
      warn "→ Either all GPUs are already used, or GPU capacity/queue quotas are too small."
      CRITICAL_ERRORS=$((CRITICAL_ERRORS + 1))
    fi
  fi
else
  warn "No ray-head pod found yet."
  info "Pods will be created when you run the training notebook."
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
echo "=================================================="
echo "📊 PREREQUISITE CHECK SUMMARY"
echo "=================================================="

if [ $CRITICAL_ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
  echo "✅ ALL CHECKS PASSED!"
  echo ""
  echo "🎉 Your environment is ready for Ray training!"
  echo ""
  echo "Next steps:"
  echo "  1. Open your notebook in OpenShift AI Dashboard"
  echo "  2. Run Module 06b notebook cells in order"
  echo "  3. Submit training job to Ray cluster"
  
elif [ $CRITICAL_ERRORS -eq 0 ] && [ $WARNINGS -gt 0 ]; then
  echo "⚠️  CHECKS PASSED WITH $WARNINGS WARNING(S)"
  echo ""
  echo "Your environment should work, but there are some warnings."
  echo "Review the warnings above and fix if needed."
  echo ""
  echo "You can proceed with training, but results may vary."
  
elif [ $CRITICAL_ERRORS -gt 0 ]; then
  echo "❌ CHECKS FAILED: $CRITICAL_ERRORS CRITICAL ERROR(S), $WARNINGS WARNING(S)"
  echo ""
  echo "🛠️  FIX THESE ISSUES BEFORE PROCEEDING:"
  echo ""
  
  if [ "$RAY_STATE" != "Managed" ] || [ "$KUEUE_STATE" != "Managed" ]; then
    echo "  📌 Platform Configuration:"
    [ "$RAY_STATE" != "Managed" ] && echo "     • Enable Ray in DataScienceCluster"
    [ "$KUEUE_STATE" != "Managed" ] && echo "     • Enable Kueue in DataScienceCluster"
    echo ""
  fi
  
  echo "  📌 Apply configuration files in order:"
  echo "     1. oc apply -f 01_gpu_flavor_and_queue.yaml"
  echo "     2. oc apply -f 02_ray_localqueue_and_cluster.yaml"
  echo "     3. oc apply -f 03_rbac_notebook_ray.yaml"
  echo ""
  echo "  📌 Wait 2-5 minutes for resources to be created"
  echo ""
  echo "  📌 Then re-run this script: ./check-ray-prereqs.sh $NS"
  
fi

echo ""
echo "=================================================="
echo "For detailed troubleshooting, see Module 06b documentation."
echo "=================================================="
