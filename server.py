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
    table_heading = ['Preview', 'URL', 'Snippet', 'MatchDateTime', 'Station', 'Show', 'DocID', 'RowID']
    query_result, num_relevant = searchEngine.run_query(query_text, num_results)
    query_time = time.time() - start_time
    return render_template('results.html', query=query_text, heading=table_heading, result=query_result, query_time=round(float(query_time), 3), num_relevant=num_relevant)

if __name__ == '__main__':
    app.run()
