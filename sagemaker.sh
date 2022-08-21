set -e
ENVIRONMENT=python3
NOTEBOOK_FILE="/home/ec2-user/SageMaker/smlambdaworkshop/training/sms_spam_classifier_mxnet.ipynb"
AUTO_STOP_FILE="/home/ec2-user/SageMaker/auto-stop.py"

source /home/ec2-user/anaconda3/bin/activate "$ENVIRONMENT"

nohup jupyter nbconvert  --to notebook --inplace --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 --execute "$NOTEBOOK_FILE" &


source /home/ec2-user/anaconda3/bin/deactivate
# PARAMETERS
IDLE_TIME=600 # 10minute

wget https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples/master/scripts/auto-stop-idle/autostop.py

(crontab -l 2>/dev/null; echo "*/1 * * * * /usr/bin/python $PWD/autostop.py --time $IDLE_TIME --ignore-connections") | crontab -