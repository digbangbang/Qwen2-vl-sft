import os
import sys
import json

def parse(string, key):
   AFO_RESOURCE_CONFIG = os.environ.get("AFO_RESOURCE_CONFIG")
   resource_json_obj = json.loads(AFO_RESOURCE_CONFIG)
   gpu_count = resource_json_obj['worker']['gpu']

   cluster_spec_json_obj = json.loads(string)
   worker_list = cluster_spec_json_obj[key]

   ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
   hostfile_path = os.path.join(ROOT_DIR, "hostfile")

   with open(hostfile_path, 'w') as f:
       for worker in worker_list:
           f.write(worker +' slots=' + str(gpu_count) + '\n')

   return hostfile_path

if __name__ == "__main__":
   parse(sys.argv[1], sys.argv[2])
