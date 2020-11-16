from flask import Flask, render_template, request
import time
import searchEngine

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    start_time = time.time()
    query_text = request.form['query']
    num_results = int(request.form['num_results'])
    table_heading = ['Preview', 'URL', 'Snippet', 'MatchDateTime', 'Station', 'Show']
    query_result = searchEngine.run_query(query_text, num_results)
    query_time = time.time() - start_time
    print(query_result)
    return render_template('results.html', query=query_text, heading=table_heading, result=query_result, query_time=round(float(query_time), 3))

if __name__ == '__main__':
    app.run(debug=True)
