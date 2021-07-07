#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Example call:

  python generate_static_html_for_visualisation.py /srv/scratch6/kew/lm_data/rrgen_de/gpt2small/210628/example_SRC_DataEditedDecoded-bs5_rep3.jsonl >| /srv/scratch6/kew/lm_data/rrgen_de/gpt2small/210628/example_SRC_DataEditedDecoded-bs5_rep3.html

"""

import sys
import json
import re
import html

html_top = """
<!DOCTYPE html>
<meta name="robots" content="noindex">
<html>
<head>
<script src="https://code.jquery.com/jquery-1.7.2.js"></script>
  <meta charset="utf-8">
  <title>JS Bin</title>
<style id="jsbin-css">

body {
  font-family: Arial, Helvetica, sans-serif
}

.hallucination {
  background-color:rgb(208, 116, 116)
}

.orange {
  background-color:rgb(201, 165, 99)
}

.green {
  background-color:rgb(115, 167, 115)
}

#data td, #data th {
  border: 2px #ddd;
  padding: 12px;
}

#data tr:nth-child(even){background-color: #f2f2f2;}

#data tr:hover {background-color: #ddd;}

#data th {
  padding-top: 12px;
  padding-bottom: 12px;
  text-align: left;
  background-color: #33BEFF;
  color: black;
}

</style>

<script id="jsbin-javascript">

// Get value on button click and show alert
// $("#submit_button").click(function () {
//   var user_threshold = $("#user_threshold").val();
//   // alert(user_threshold);
//   return user_threshold
// });

function display_highlights() {
  var user_threshold = $("#user_threshold").val()
  var score;
  var th;

  console.log(user_threshold)

  $('span').each(function (index) {

    th = $(this);

    score = parseFloat($(this).attr('score'));
    console.log(score);

    if (score >= user_threshold) {
      th.addClass('hallucination');
    } else {
      th.removeClass('hallucination')
    };
  });
};

$(document).ready(function () {
  display_highlights()
});

</script>


</head>
<body>
  
<div id="description">
<h2>Hallucination Detection</h2>
<p>
  This aims to visualise reference-free token-level
  hallucination for text-to-text generation.
</p>

<p>
  Select a threshold and inspect the highlighted data to
  find a suitable threshold value.
</p>

<form>
  <label for="user_threshold">Threshold:</label>
  <input type="text" id="user_threshold" name="user_threshold" value="0.9">
  <button type="button" id="submit_button" onclick="display_highlights()">Submit</button>
</form>
<br/><br/>
</div>

<table id="data">
	<tbody>
    <tr>
      <th> hal score </th>
      <th> hypothesis </th>
      <th> source text </th>
    </tr>

"""

html_bottom = """
  </tbody>
</table>
</body>
</html>
"""


def iter_lines(infile):
    with open(infile, 'r',encoding='utf8') as inf:
        for line in inf:
            yield json.loads(line.strip())


def get_simple_html(data):
    """
    converts data entry (dict) to html table row string

    <tr>
      <td> hallu score </td>
      <td> hypothesis </td>
      <td> source text </td>
    </tr>

    """

    tokens = ['<div>']
    for score, token in zip(data['diff_pos_scores'], data['target_tokens'].split()):
        # html_str = f'<span score={score}>{token.replace("▁", " ")}</span>'
        html_str = f'<span score={score}>{token} </span>'
        tokens.append(html_str)
    tokens.append('</div>')
    
    # escape html
    src_text = html.escape(data['source_text'])

    # return ''.join(tokens)
    table_row = """
      <tr>
        <td>{:.3f}</td>
        <td>{}</td>
        <td>{}</td>
      </tr>""".format(
        data['hal-lm_score'], 
        ''.join(tokens),
        src_text
      )


    return table_row

if __name__ == '__main__':
    infile = sys.argv[1]
    print(html_top)
    for data in iter_lines(infile):
        print(get_simple_html(data))
    print(html_bottom)