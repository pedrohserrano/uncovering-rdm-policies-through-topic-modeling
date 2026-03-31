This folder contains the raw policy source data used during the project.

**data_sources.csv** logs which universities were considered and which link houses the RDM policy.  
**pdf/** contains the RDM policies in PDF format.  
**markdown/** contains the MinerU-extracted markdown versions of those policies. The extraction was done online using MinerU Version 2025/01/22 1.1.0: https://huggingface.co/spaces/opendatalab/MinerU . For local extraction, see: https://mineru.readthedocs.io/en/latest/index.html .  

Processed tabular inputs and outputs are stored separately under **data/**:

- **data/input/** for shared pipeline inputs such as `documents.csv` and `documents_cleaned.csv`
- **data/output/** for chunking, labeling, and analysis outputs
