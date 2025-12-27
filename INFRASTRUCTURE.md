# Infrastructure Implementation Guide

This document outlines the steps to implement the Global Load Balancing and CDN infrastructure required for high performance in Asia.

## Prerequisites

- Google Cloud CLI (`gcloud`) installed and authenticated.
- Project ID set: `vanityforge`.
- Existing VM Instance Name: `vanity-vm` (Replace with your actual VM name).
- Zone: `us-central1-a` (Replace with your actual zone).

## 1. Reserve a Global Static IP

We need a stable global IP address for the Load Balancer.

```bash
gcloud compute addresses create vanity-global-ip \
    --global \
    --ip-version=IPV4
```

Retrieve the IP address:
```bash
gcloud compute addresses describe vanity-global-ip --global --format="get(address)"
```

## 2. Prepare the VM Instance Group

Since we are pointing to a VM instance, we must wrap it in an Unmanaged Instance Group.

1. **Create an Unmanaged Instance Group:**
   ```bash
   gcloud compute instance-groups unmanaged create vanity-ig \
       --zone=us-central1-a
   ```

2. **Add the VM to the Instance Group:**
   ```bash
   gcloud compute instance-groups unmanaged add-instances vanity-ig \
       --zone=us-central1-a \
       --instances=vanity-vm
   ```

3. **Set Named Ports:**
   The Load Balancer needs to know which port the application listens on (Flask binds to 8080).
   ```bash
   gcloud compute instance-groups unmanaged set-named-ports vanity-ig \
       --zone=us-central1-a \
       --named-ports=http:8080
   ```

## 3. Create a Backend Service (Enable Cloud CDN)

This is where we enable the CDN and attach the Instance Group.

1. **Create the Backend Service:**
   ```bash
   gcloud compute backend-services create vanity-backend \
       --global \
       --protocol=HTTP \
       --port-name=http \
       --timeout=30s \
       --enable-cdn \
       --cache-mode=CACHE_ALL_STATIC \
       --default-ttl=3600
   ```

2. **Add the Instance Group to the Backend Service:**
   ```bash
   gcloud compute backend-services add-backend vanity-backend \
       --global \
       --instance-group=vanity-ig \
       --instance-group-zone=us-central1-a \
       --balancing-mode=UTILIZATION \
       --max-utilization=0.8
   ```

## 4. Health Check

The Load Balancer needs a health check to know if the VM is responsive.

1. **Create a Health Check:**
   ```bash
   gcloud compute health-checks create http vanity-health-check \
       --port=8080 \
       --request-path="/"
   ```

2. **Attach Health Check to Backend Service:**
   ```bash
   gcloud compute backend-services update vanity-backend \
       --global \
       --health-checks=vanity-health-check
   ```

3. **Firewall Rule:**
   Allow Google Load Balancer IP ranges to access your VM.
   ```bash
   gcloud compute firewall-rules create allow-lb-health-check \
       --network=default \
       --action=allow \
       --direction=ingress \
       --source-ranges=130.211.0.0/22,35.191.0.0/16 \
       --target-tags=http-server \
       --rules=tcp:8080
   ```
   *Note: Ensure your VM has the tag `http-server`.*

## 5. SSL Certificate

Create a Google-managed SSL certificate for the domain.

```bash
gcloud compute ssl-certificates create vanity-ssl-cert \
    --domains=vanityforge.org
```

## 6. URL Map (Load Balancer)

Create the URL map that routes incoming requests to the backend service.

```bash
gcloud compute url-maps create vanity-lb \
    --default-service=vanity-backend
```

## 7. Target HTTPS Proxy

Connect the URL map and SSL certificate.

```bash
gcloud compute target-https-proxies create vanity-https-proxy \
    --ssl-certificates=vanity-ssl-cert \
    --url-map=vanity-lb
```

## 8. Global Forwarding Rule

Finally, expose the Load Balancer on the reserved Global IP.

```bash
gcloud compute forwarding-rules create vanity-forwarding-rule \
    --global \
    --target-https-proxy=vanity-https-proxy \
    --address=vanity-global-ip \
    --ports=443
```

## 9. DNS Update

Update your DNS A record for `vanityforge.org` to point to the IP address allocated in Step 1.

## Verification

Once deployed, the Load Balancer will:
1. Terminate SSL globally.
2. Serve cached content (`index.html`, `locales/zh-CN.json`, `assets/*`) from the edge location nearest to the user (e.g., Tokyo, Hong Kong).
3. Route dynamic API requests (`/api/*`) directly to the VM instance in `us-central1`, bypassing the cache as configured in `vm_server.py`.
