# Lambdas development

## Setup

### One time setup
1. setup venv `python3 -m venv venv`
1. Run `python3 -m pip install -r requirements.txt`
1. Run `python3 -m pip install -e .`

### To execute

The local commands write generated HTML to `/tmp/storyweb/index.html` 

1. run `storyweb <command>` i.e.  `storyweb read` to generate HTML file
1. run `cd /tmp/storyweb && python3 -m http.server`
1. navigate to `localhost:8000`

Install some plugin for active reload for simpler development.


### To deploy

Do these steps for each modified lambda:

1.  cd venv/lib/python3.8/site-packages
2.  zip -r ../../../../my-deployment-package.zip
3.  cd ../../../../lambdas
4.  zip -g -r my-deployment-package.zip storyweb
5.  cp storyweb/lambdas/read_story_lambda/read_story_lambda.py /tmp/storyweb/lambda_function.py
6.  zip -g -j my-deployment-package.zip /tmp/storyweb/lambda_function.py 
7.  upload to the correct lambda