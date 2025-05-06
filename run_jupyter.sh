#!/bin/bash
# Description: Launch Jupyter Lab on HPC and print SSH tunnel instructions.

# --- Set critical variables
readonly IPADDRESS=$(hostname -I | tr ' ' '\n' | grep '10.211.4.')
readonly PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# --- Print user instructions
cat <<END
1. SSH tunnel from your workstation:

   Linux/Mac:
   ssh -NL 8890:${HOSTNAME}:${PORT} ${USER}@hpc.create.kcl.ac.uk

   Windows (PowerShell):
   ssh -m hmac-sha2-512 -NL 8888:${HOSTNAME}:${PORT} ${USER}@hpc.create.kcl.ac.uk

   Browser URL:
   http://localhost:8890/lab?token=<PASTE_TOKEN_FROM_OUTPUT_BELOW>

To terminate the notebook later:
   scancel -f ${SLURM_JOB_ID}
END

# --- Launch Jupyter Lab
jupyter lab --port=${PORT} --ip=${IPADDRESS} --no-browser --notebook-dir=/app/NBT-Classifier
