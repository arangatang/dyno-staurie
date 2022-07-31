#/bin/bash

LAMBDA=$1
DESTINATION=$2
ROOT=$(pwd)
OUTPUT_DIR=/tmp/storyweb
OUTPUT_ZIP=${OUTPUT_DIR}/${LAMBDA}.zip
mkdir -p $OUTPUT_DIR
cd ${ROOT}/lambdas/venv/lib/python3.8/site-packages
zip -r -q $OUTPUT_ZIP .
cd $ROOT/lambdas
zip -g -r -q $OUTPUT_ZIP storyweb
cp storyweb/lambdas/${LAMBDA}/${LAMBDA}.py /tmp/storyweb/lambda_function.py
zip -g -j $OUTPUT_ZIP /tmp/storyweb/lambda_function.py
cd $ROOT
echo $LAMBDA
echo $DESTINATION
aws lambda update-function-code --function-name $DESTINATION --zip-file fileb://$OUTPUT_ZIP --publish