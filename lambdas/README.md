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