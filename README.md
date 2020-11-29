# Text Summerization using Pointer Generator Network

Steps to run the pertained model on the dataset

1). Download the finished dataset from the link shared below.
    So here is the direct link where i have mentioned the link from where i have downloaded the final prepared data which will be used by us https://github.com/abisee/cnn-dailymail

2). The model which is shared in the above link is based on python 2 so in our case we need to use the python 3 based model..which I downloaded from https://github.com/becxer/pointer-generator/

There were multiple bugs which are resolve in this repository & modification which I did on it. Attaching the zip folder for the both the models(with & witout coverage) so that you guys can download it straight away.

3). You need to download the pretrained model from
https://drive.google.com/drive/folders/1_MHev5OvCIw2q8q44P43zUh9Cmh3LU0u?usp=sharing

Please remember there are 2 pretrained models which are available. The above link has model which is based on tensor-flow 1.15.0. There are few deprecated api’s but it will work.

4). Download the finished_files from the below link 
https://drive.google.com/drive/folders/1WsnQ7o6uMRjY4OQai0gpMY50E3nQgrbr?usp=sharing

It is mentioned at the https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail the link from where we can download the preprocessed data.

Now keep all the folders in the same place.

Then go in to pointer-generator folder and then run the below command.

python run_summarization.py --mode=decode 
--data_path=/home/ujwal/Documents/Text_Summarizer/finished_files/finished_files/chunked/val_* 
--vocab_path=/home/ujwal/Documents/Text_Summarizer/finished_files/finished_files/vocab 
--log_root=/home/ujwal/Documents/Text_Summarizer/logs/testExperiment_1/train 
--exp_name=myexperiment_1 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 –single_pass=1

You need to modify the path according to you folder structure.

Run it and you will get the output in the pretrained_model folder.

Now Lets see how can we install the pyrouge and get the result


----------------------------
To get the evaluation done
----------------------------

pip uninstall pyrouge
git clone https://github.com/bheinzerling/pyrouge
cd pyrouge
pip install -e .
pyrouge_set_rouge_path </absolute/path/to/ROUGE-1.5.5/directory>
python -m pyrouge.test

##############################################################################################################################
----------------
create eval.py
----------------

from pyrouge import Rouge155
r = Rouge155()
# set directories
r.system_dir = "/home/kunal/Documents/Text_Summarizer/logs/testExperiment_1/decode_val_400maxenc_4beam_35mindec_100maxdec_ckpt-50064/decoded/"
r.model_dir = "/home/kunal/Documents/Text_Summarizer/logs/testExperiment_1/decode_val_400maxenc_4beam_35mindec_100maxdec_ckpt-50064/reference/"

#define the patterns
r.system_filename_pattern = "(\d+)_decoded.txt"
r.model_filename_pattern = "#ID#_reference.txt"

#use default parameters to run the evaluation
output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)
______________
python -m eval
______________

################################################################################################################################

Please remember we need to copy the rouge-1.5.5 and keep it in pyrouge folder.

If you don’t have then download the pyrouge from https://github.com/andersjo/pyrouge and copy ROUGE-1.5.5 from tools folder and past it in the above pyrouge don’t use this downloaded pyrouge..I faced lot of problem with it.


####How to create the data set####
Please change the script file "make_datafiles.py" in cnn-dailymail and then run it with python3.Present in the main folder.
It will create the story and bin files.As we had lot of data and becuase of lack of resources we have modified the script to get the bin files only for the cnn.

####Fuure Work####
1) Would create a docker image for the training and evaluation process.
2) Would upgrade the code to tensorflow 2 and above for less compatibility issues.
3) Hyper-parameter tuning the model for better results.

