# PERO
Reference code for Findings of [Findings of ACL 2021](https://2021.aclweb.org/) paper -  **Reordering Examples Helps during Priming-based Few-Shot Learning**.

<p align="center">
  <img align="center" src="https://github.com/SawanKumar28/pero/blob/main/images/arch.jpg" alt="...">
</p>

## Dependencies
The code was written with, or depends on:
* Python 3.6
* Pytorch 1.7.0
* Transformers 3.4.0
## Running the code
1. Create a virtualenv and install dependecies
      ```bash
      virtualenv -p python3 env
      source env/bin/activate
      pip3 install -r requirements.txt
      ``` 
1. Download data following instructions at https://github.com/ucinlp/autoprompt  [2] and unzip in the same folder as this repository. For fact-retreival experiments, download LAMA [3] data from https://github.com/facebookresearch/LAMA and copy the ```relations.jsonl``` file into ```data/fact-retrieval/original/```.
1. To Run classification tasks using 10 training examples: <dataset> can be sst2 or sicke2b. <mode> can be pero or pero_abl (without sep token learning). <start_idx> specifies the trainin split, to reproduce results from the paper, use 0,10,20,30,40.
      ```bash
      bash run_clf.sh 0 <dataset> <mode> <start_idx> ./saved_models/outputdir1
      ```   
1. Similarly, to run fact retrieval task:
      ```bash
       bash run_fact_retrieval.sh 0 <mode> <start_idx> saved_models/outputdir2
      ```   
  
## Citation
If you use this code, please consider citing:
      
[1] Sawan Kumar and Partha Talukdar. 2021. Reordering Examples Helps during Priming-based Few-Shot Learning. To appear in Findings of ACL, 2021. Association for Computational Linguistics.
  
## References
  
[2] Shin, Taylor, et al. "Eliciting Knowledge from Language Models Using Automatically Generated Prompts." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020.
  
[3] Petroni, Fabio, et al. "Language Models as Knowledge Bases?." Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2019.
  
## Contact
For any clarification, comments, or suggestions please create an issue or contact sawankumar@iisc.ac.in
