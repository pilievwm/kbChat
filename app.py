from flask import Flask, request
from functools import wraps
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')

def token_required(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token or token.split()[1] != ACCESS_TOKEN:
            return {'message': 'Unauthorized access'}, 401

        return func(*args, **kwargs)

    return decorated

@app.route('/query', methods=['POST'])
@token_required
def greeting():
    question = request.json.get('question')
    import botAnswer
    response = botAnswer.answer_bot(question)
    return response

@app.route('/search', methods=['POST'])
@token_required
def finding():
    search = request.json.get('search')
    import searchBot
    response = searchBot.answer_bot(search)
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
