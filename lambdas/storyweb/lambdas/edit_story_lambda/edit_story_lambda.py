from typing import Dict, List
import boto3
from storyweb.utils.chapters import Chapter
from storyweb.utils.constants import DDB_TABLE_NAME, error_page
from storyweb.style import load_css
from storyweb.utils.loaders import get_jinja_environment
from storyweb.utils.utils import from_dynamodb_to_json

ddbclient = boto3.client("dynamodb")


def lambda_handler(event, context):
    chapter_id = event.get("queryStringParameters", {}).get("chapter", "1")
    story_id = event.get("queryStringParameters", {}).get("story", "test_story")

    if not chapter_id or not story_id:
        return error_page()
        # chapter_id = "3"
        # story_id = "test_story"

    print(f"Querying for story: {story_id}")
    # response = ddbclient.query(
    #     TableName=DDB_TABLE_NAME,
    #     KeyConditionExpression="story = :story",
    #     ExpressionAttributeValues={":story": {"S": story_id}},
    # )
    
    response = RESPONSE
    print("Got response")
    print(response)

    items = response.get("Items", [])
    if len(items) == 0:
        return error_page()

    chapters = {}

    for item in items:
        chapter = Chapter(chapter=from_dynamodb_to_json(item))
        chapters[chapter.chapter_id] = chapter

    nodes = get_nodes(chapters)
    edges = get_edges(chapters)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/html"},
        "body": get_jinja_environment()
        .get_template("edit_story_template.jinja")
        .render(
            nodes=nodes,
            edges=edges,
            select_node=chapter_id if chapter_id else "",
            chapters=[chapter.to_dict() for chapter in chapters.values()],
            story_name=story_id,
            new_option_id=max([int(id.split(".")[0]) for id in chapters]) + 1,
        ),
    }


def get_nodes(chapters: Dict[str, Chapter]):
    def get_node(id: str, is_work_in_progress: bool, is_final: bool, is_start: bool):
        if is_work_in_progress:
            node_color = "#FFFF00"  # yellow
        elif is_start:
            node_color = "#00FF00"  # green
        elif is_final:
            node_color = "#FF0000"  # red
        else:
            node_color = "#4DA4EA"  # blue

        color = 'color: { background: "%s"}' % node_color
        return '{ id: "%s", label: "%s", %s },' % (id, id, color)

    nodes = []
    for key, chapter in chapters.items():
        nodes.append(
            get_node(
                key, chapter.is_unfinished(), chapter.is_final(), chapter.is_start()
            )
        )

    return "\n".join(nodes)


def get_edges(chapters: Dict[str, Chapter]):
    def get_edge(_from, to):
        return (
            '{ from: %s, to: %s, arrows: { to: { enabled: true, type: "arrow" }}},'
            % (
                _from,
                to,
            )
        )

    edges = []
    for chapter_id, chapter in chapters.items():
        for choice in chapter.get_choices().keys():
            edges.append(get_edge(chapter_id, choice))
    return """
    """.join(
        edges
    )




RESPONSE = {'Items': [{'chapter': {'S': '-1'}, 'options': {'L': [{'M': {'next': {'S': '5'}, 'text': {'S': 'The boy was confused, the dream had felt so real...'}}}]}, 'text': {'S': 'The boy saw that the old oak tree had fallen over and inside sat a small pixie with purple eyes. Then the boy woke up.'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/FINAL_CHAPTER-2.png'}}, {'chapter': {'S': '1'}, 'is_start_node': {'BOOL': True}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/1-2.png'}, 'options': {'L': [{'M': {'next': {'S': '1.1'}, 'text': {'S': 'But was he a boy upon that time?'}}}, {'M': {'next': {'S': '11'}, 'text': {'S': 'He was a martian...'}}}, {'M': {'next': {'S': '13'}, 'text': {'S': 'Actually he was Adam'}}}, {'M': {'next': {'S': '2'}, 'text': {'S': 'Continue'}}}, {'M': {'next': {'S': '18'}, 'text': {'S': 'He loved liqourice '}}}]}, 'text': {'S': 'Once upon a time there was a young boy.'}, 'story': {'S': 'test_story'}, 'disabled_options': {'L': [{'S': '13'}, {'S': '18'}]}}, {'chapter': {'S': '1.1'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/1.1-0.png'}, 'text': {'S': 'The boy transformed into a mermaid and jumped into a nearby lake.'}, 'options': {'L': [{'M': {'next': {'S': '10'}, 'text': {'S': 'He swam down to visit his family'}}}]}, 'story': {'S': 'test_story'}, 'disabled_options': {'L': [{'S': '10'}]}}, {'chapter': {'S': '10'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/10-0.png'}, 'text': {'S': 'His family was living in a large underwater sand castle which was guarded by mean looking seahorses.'}, 'options': {'L': [{'M': {'next': {'S': '12'}, 'text': {'S': 'He went to say hi to the neighbours '}}}]}, 'story': {'S': 'test_story'}, 'disabled_options': {'L': [{'S': '12'}]}}, {'chapter': {'S': '11'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/11-0.png'}, 'text': {'S': 'The boy lived in a tiny glass sphere on one of the moons on mars.'}, 'options': {'L': [{'M': {'next': {'S': '17'}, 'text': {'S': 'He lived on the smallest most desolate moon of them all.'}}}]}, 'story': {'S': 'test_story'}, 'disabled_options': {'L': [{'S': '17'}]}}, {'chapter': {'S': '12'}, 'text': {'S': 'The neighbours were ducks, who was hiding from the plague of the land known as the five crows'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/12-0.png'}}, {'chapter': {'S': '13'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/13-4.png'}, 'text': {'S': 'Adam likes to dress as a slutty nurse'}, 'options': {'L': [{'M': {'next': {'S': '14'}, 'text': {'S': 'A patient rolled into the hospital'}}}, {'M': {'next': {'S': '15'}, 'text': {'S': 'A man lies sleeping in the hospital bed'}}}, {'M': {'next': {'S': '16'}, 'text': {'S': 'test'}}}]}, 'story': {'S': 'test_story'}, 'disabled_options': {'L': [{'S': '14'}, {'S': '15'}, {'S': '16'}]}}, {'chapter': {'S': '14'}, 'text': {'S': 'The person had broken legs and layed on a hospital stroller'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/14-0.png'}}, {'chapter': {'S': '15'}, 'text': {'S': 'Closely examining the guy in the bed you see a tent on his crotch.'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/15-0.png'}}, {'chapter': {'S': '16'}, 'text': {'S': 'Adam liked to dress as a slutty nurse when going to parties and parades! He liked wearing a lot of makeup and that made him look younger, but he was a sweetheart and didn\'t mind. He loved the attention he got from women, especially from his girlfriends. Adam enjoyed sex, and he loved to please his partner. It wasn\'t unusual for Adam to be completely turned on at a time when he felt he had no one in his life. If he wanted something, he just went right to the source. As a result, Adam was very selective about who he let into his apartment and what he did with them. When Adam couldn\'t satisfy - and had too much - his girlfriend, she would retreat to her bedroom. While Adam stayed on the couch and waited for her to come back to him, the two of them watched television. Sometimes he would fall asleep watching television on a couch or while reading, or just sitting on his bed with a magazine. Once, a young girl came into Adam\'s apartment while he and his girl were watching TV. She was so beautiful that Adam tried one last, desperate attempt. "I\'m going out with her," he told her. Her eyes widened, then she ran out of his bedroom and he heard her crashing around the house. In his anger, his phone rang. Was she there? He left it on. Did she even know it rang? Adam called again. Then he finally gave up. There was no answer. So he went to bed. A few days later, that girl came back into my apartment. We were still watching tv. Because of how she had left, I knew she was not someone I should have trusted. Since I had the apartment bugged, all I heard was her screaming. (I have the audio from that phone call on my desk. Probably she would not remember me if I called her back and asked her.) Later, we were invited to a party, where we drank and talked about the girl from the first apartment all night. Was that person still there? I asked. Yes, he said. Well, how did he find me? He would often ask if the person was still there when they answered his door. Adam had learned that people didn\'t like to admit they weren\'t the one they thought they were. There were usually a lot of people present, too many for me to count. Every night, after he passed out, he asked if there was anyone there he could speak to. (I assume he had to see if they wanted to play a game of truth or dare or what have you. Perhaps they only wanted their mother to know they still lived. Or they knew that their lover had moved on, even though it hadn\'t been the same person. One day, while Adam and I were talking about our first date, after a night of drinking, my boyfriend said that when we first met, our conversation was so fast that it made us seem so much more alike, like we could talk to each other like that . He was right. We became fast friends and soon moved to the same apartment complex. Whenever we went out, he took me to his place for a drink. On the way home, I asked him if he liked sex as much as I did. Not really, just because I never knew , yet he laughed and said he was afraid of it at first, because it felt like he wasn\'t doing enough. Of course, once he got over that fear, then he really liked it! After a while of talking to me about sex - the pleasure he felt from it, how much he enjoyed it, even the stories he told, - he started asking me what I thought about it too. "I never really had the chance to tell you that before, either." I said. \' And so began our courtship. We started dating the summer after we met. Every day , Adam would come to my place to play with his new toys, as if we weren`t already dating, or just to let me know he had arrived. When I first moved in, Adam was there first thing. Then my roommate. Soon, everyone was in my house. Everyone. In , a red-headed girl would pick us up from school, ask us how our day was, tell us to smile, then open the back door for Adam to go in for a quickie. She would tease him a little bit with her hand, making sure he would get a good look at his backside. Before our final night together, Adam called me on my cell phone from a different phone, saying he could not wait for me to hear his voice. On our last night, he came to our apartment to tell me that he loved me a thousand times and had sex with me at my apartment, on my couch. Of course I knew he knew we were having sex, that was why he did that, we agreed that we had done it in the past, yet it wasn` t until a month later, when he asked me to marry him and tell him how I felt, that my heart skipped a beat. He was on fire. When I told him he could not have me because he wanted to, I really didn` d know what to say. "I thought that you loved him?" he said. No, of course you didn `t, why would you, Adam laughed, that was the first thing he said to that! It was too bad my girlfriend didn t want me to! This is where I stopped talking. You would think, if I had the right words, there was no problem - and that is not the case. Because there is. His words started me thinking. He said I should tell him what it meant. That it could mean anything, anything. Then I remembered. What it really meant was, we were just different. We were not two beings, two people, trying to find each others` soul. If he really wanted'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/16-0.png'}}, {'chapter': {'S': '17'}, 'text': {'S': 'This small moon is Deimos, the son of the god of war Ares.'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/17-0.png'}}, {'chapter': {'S': '18'}, 'text': {'S': 'Especially liqourice lollipops'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/18-0.png'}}, {'chapter': {'S': '2'}, 'options': {'L': [{'M': {'next': {'S': '3'}}}]}, 'text': {'S': 'The boy lived in a small house in a large forest'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/2-7.png'}}, {'chapter': {'S': '3'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/3-3.png'}, 'options': {'L': [{'M': {'next': {'S': '3.1'}, 'text': {'S': 'The boy went outside to investigate'}}}, {'M': {'next': {'S': '3.2'}, 'text': {'S': 'The boy, who was used to these kind of sounds, quickly fell back asleep'}}}, {'M': {'next': {'S': '4'}, 'text': {'S': 'The boy was so scared that he hid underneath his blanket!'}}}]}, 'text': {'S': 'One night, the boy woke up from a loud cracking sound just outside his cute little orange house.'}, 'story': {'S': 'test_story'}, 'disabled_options': {'L': [{'S': '4'}]}}, {'chapter': {'S': '3.1'}, 'options': {'L': [{'M': {'next': {'S': '-1'}, 'text': {'S': 'he fetched a flashlight inside, and then returned'}}}, {'M': {'next': {'S': '3.2'}, 'text': {'S': 'he went back to bed to wait for the morning light'}}}]}, 'text': {'S': 'The night was dark so the boy could not see anything.'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story_1/images/3.1-2.png'}}, {'chapter': {'S': '3.2'}, 'options': {'L': [{'M': {'next': {'S': '-1'}, 'text': {'S': 'The boy had some breakfast and then went outside to check what had made the sound'}}}, {'M': {'next': {'S': '3.2'}, 'text': {'S': 'he stayed in bed cause he was lazy'}}}]}, 'text': {'S': 'A few hours later it was raining cats and dogs.'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/3.2-0.png'}}, {'chapter': {'S': '4'}, 'options': {'L': [{'M': {'next': {'S': '-1'}, 'text': {'S': 'in the morning he went outside, to check what had happened...'}}}]}, 'text': {'S': 'The boy hid under the fluffy green duvet.'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/4-0.png'}}, {'chapter': {'S': '5'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/5-0.png'}, 'options': {'L': [{'M': {'next': {'S': '5'}, 'text': {'S': 'He sat down on a stub for a while.'}}}, {'M': {'next': {'S': '6'}, 'text': {'S': 'The boy went on with his day, he went on chopped wood, picked berries and sometimes jumped around and sang. '}}}, {'M': {'next': {'S': '7'}, 'text': {'S': 'He went back to sleep.'}}}]}, 'text': {'S': 'Unsure of what to do next, the boy pondered what he would do. He was a bit tired, but had so many things he needed to do!'}, 'story': {'S': 'test_story'}, 'disabled_options': {'L': [{'S': '6'}, {'S': '7'}]}}, {'chapter': {'S': '6'}, 'text': {'S': 'Purple graph with nodes in varying different colors in the style of a photograph'}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/6-0.png'}}, {'chapter': {'S': '7'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/7-0.png'}, 'options': {'L': [{'M': {'next': {'S': '3.1'}, 'text': {'S': 'He woke up startled, by a noise outside...'}}}, {'M': {'next': {'S': '8'}, 'text': {'S': 'test choice'}}}]}, 'text': {'S': 'His dreams were wild, the purple eyes of the pixie was everywhere. '}, 'story': {'S': 'test_story'}, 'disabled_options': {'L': [{'S': '8'}]}}, {'chapter': {'S': '8'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/8-0.png'}, 'text': {'S': 'The purple pixie said that he was too small to live in the forest alone'}, 'options': {'L': [{'M': {'next': {'S': '9'}, 'text': {'S': 'The boy then told a story'}}}]}, 'story': {'S': 'test_story'}, 'disabled_options': {'L': [{'S': '9'}]}}, {'chapter': {'S': '9'}, 'text': {'S': 'Once upon a time the boy lived with the mermaids in the nearby pond. They lived in a castle far far down in peace with the dolphins and crabs. '}, 'story': {'S': 'test_story'}, 'image': {'S': 'https://story-images.s3.eu-central-1.amazonaws.com/stories/test_story/images/9-2.png'}}], 'Count': 22, 'ScannedCount': 22, 'ResponseMetadata': {'RequestId': '1K27D593P5M45P7CG77H3EN2IRVV4KQNSO5AEMVJF66Q9ASUAAJG', 'HTTPStatusCode': 200, 'HTTPHeaders': {'server': 'Server', 'date': 'Fri, 05 Aug 2022 16:21:56 GMT', 'content-type': 'application/x-amz-json-1.0', 'content-length': '13224', 'connection': 'keep-alive', 'x-amzn-requestid': '1K27D593P5M45P7CG77H3EN2IRVV4KQNSO5AEMVJF66Q9ASUAAJG', 'x-amz-crc32': '4289263495'}, 'RetryAttempts': 0}}