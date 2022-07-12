DDB_TABLE = "arn:aws:dynamodb:eu-central-1:679585051464:table/stories"
DDB_TABLE_NAME = "stories"
LAST_CHAPTER_PREFIX = "FINAL_CHAPTER"


STYLE = """
body {
    background-color: white;
}

img {
    display: block;
    width: 500px;
    margin-left: auto;
    margin-right: auto;
    margin-top: 60px;
}

p {
    text-align: center;
    line-height: 100px;
    font-size: xx-large;
}

.center{
    width: 45%;
    margin: 0 auto;
    background-color: whitesmoke;
    border-style: ridge;
    border-radius: 5px
}

a:link { text-decoration: none; }
a:visited { text-decoration: none; }

.center:hover {
    background-color: floralwhite;
}

a {
    font-size: x-large;
    line-height: 50px;
}
"""


HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <title>Network</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        #mynetwork {
            width: 600px;
            height: 400px;
            border: 1px solid lightgray;
        }
    </style>
    {STYLE}
</head>

<body>
    <div id="mynetwork"></div>
    <div id="edit_windows">
        {EDIT_WINDOWS}
    </div>
    <script type="text/javascript">
        // create an array with nodes
        var nodes = new vis.DataSet([
            {NODES}
        ]);

        // create an array with edges
        var edges = new vis.DataSet([
            {EDGES}
        ]);

        // create a network
        var container = document.getElementById("mynetwork");
        var data = {
            nodes: nodes,
            edges: edges,
        };
        var options = { };
        var network = new vis.Network(container, data, options);
    </script>
</body>

</html>
""".format(
    STYLE=STYLE
)


EDIT_WINDOW = """
<div class="edit-window" id="{CHAPTER_ID}>
    <img src="{IMAGE}" loading="lazy"/>
    <form action="/process_edit_story" method="get">
        <label for="_text">Text</label>
        <input type="text" id="_text" name="_text"><br><br>
        {EXISTING_OPTIONS}
        <label for="new_option">New choice: </label>
        <input type="text" id="new_option" name="new_option"><br><br>

        <input type="text" id="story_id", name="story_id" class="hidden">
        <input type="text" id="chapter_id", name="story_id" class="hidden">

        <input type="submit" value="Submit">
      </form>
</div>
"""
