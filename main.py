from modules import *
from flask import Flask, render_template, redirect, url_for
from flask import request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    ai = AI_information()
    info = ai.info()
    rs = request.args.get('result')
    data = request.args.get('data')
    return render_template('index.html', rs=rs, data=data, info=info)


@app.route('/process', methods=['POST'])
def process():
    src_lang = "en"
    tgt_lang = "vi"
    data = request.form.get('content')
    trans = AI_Translation()
    result = trans.translate_text(input_text=data, src_lang=src_lang, tgt_lang=tgt_lang)
    return redirect(url_for('translation', result=result, data=data))
    
@app.route('/translation')
def translation():
    ai = AI_information()
    info = ai.info()
    rs = request.args.get('result')
    data = request.args.get('data')
    return render_template('translation.html', rs=rs, data=data, info=info)

if __name__ == '__main__':
    app.run(debug=True)