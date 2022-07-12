from flask import Flask

app = Flask(__name__)

@app.route("/members")
def members():
    return {
       'text': "hello there",
       'options': {
           'option_id': 'sdsad',
          },
       'image': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/1-3.png'
    }

if __name__=="__main__":
    app.run(debug=True)