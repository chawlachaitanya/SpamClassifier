import json
import re
import string
import sys
from hashlib import md5
import boto3
import email
import os

message_template = """We received your email sent at {0} with the subject {1}. Here is a 240 character sample of the email body: {2}.The email was categorized as {3} with a {4}% confidence."""

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

endpoint_name = os.environ['ENDPOINT_NAME']


def vectorize_sequences(sequences, vocabulary_length):
    results = []
    for sequence in sequences:
        results.append([0] * vocabulary_length)

    for i, sequence in enumerate(sequences):
        for position in sequence:
            results[i][position] = 1

    return results


def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]


def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    """One-hot encodes a text into a list of word indexes of size n.
    This is a wrapper to the `hashing_trick` function using `hash` as the
    hashing function; unicity of word to index mapping non-guaranteed.
    # Arguments
        text: Input text (string).
        n: int. Size of vocabulary.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        List of integers in [1, n]. Each integer encodes a word
        (unicity non-guaranteed).
    """
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    """Converts a text to a sequence of indexes in a fixed-size hashing space.
    # Arguments
        text: Input text (string).
        n: Dimension of the hashing space.
        hash_function: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of integer word indices (unicity non-guaranteed).
    `0` is a reserved index that won't be assigned to any word.
    Two or more words may be assigned to the same index, due to possible
    collisions by the hashing function.
    The [probability](
        https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
    of a collision is in relation to the dimension of the hashing space and
    the number of distinct objects.
    """
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]


def transform_data_before_sending_to_model(messages):
    vocabulary_length = 9013
    one_hot_messages = one_hot_encode(messages, vocabulary_length)
    print(type(one_hot_messages))
    encoded_messages = vectorize_sequences(one_hot_messages, vocabulary_length)
    print(type(encoded_messages))
    return json.dumps(encoded_messages)


def get_recommendation(message):
    sagemaker_runtime = boto3.client("sagemaker-runtime", region_name='us-east-1')

    data = transform_data_before_sending_to_model([message])

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=data,
        ContentType='application/json',
        Accept='string',
        CustomAttributes='string'
    )

    prediction = json.loads(response['Body'].read().decode())
    score = prediction['predicted_probability'][0][0]
    classification = prediction['predicted_label'][0][0]
    return classification, score


def get_message_body(msg):
    if not msg.is_multipart():
        return msg.get_payload()

    return get_message_body(msg.get_payload()[0])


def send_email(to, body):
    subject = 'SPAM Check'
    print(to, body)
    email_client = boto3.client('ses', region_name="us-east-1")

    try:
        response = email_client.send_email(
            Destination={
                'ToAddresses': [
                    to
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': 'UTF-8',
                        'Data': body,
                    }
                },
                'Subject': {
                    'Charset': 'UTF-8',
                    'Data': subject,
                },
            },
            Source='SpamTest@ccbd.chaitanyachawla.com',
        )
        print(response)

        return response
    except Exception as e:
        print(e)


def process_email(bucket, key):
    client = boto3.client('s3')
    ob = client.get_object(Bucket=bucket, Key=key)
    msg = email.message_from_bytes(ob['Body'].read())
    subject = msg['Subject']
    date = msg['Date']
    payload = get_message_body(msg)
    sender = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", msg['from'])[0]

    body = payload.replace('\r', '').replace('\n', ' ').strip()
    classification, score = get_recommendation(body)

    if len(body) > 240:
        body = body[:240]

    response_message = message_template.format(date, subject, body, classification, score)
    send_email(sender, response_message)


def lambda_handler(event, context):
    # TODO implement
    print(event)
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        process_email(bucket, key)
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
