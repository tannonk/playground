#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json

html_top = """
<!DOCTYPE html>
<meta name="robots" content="noindex">
<html>
<head>
<script src="https://code.jquery.com/jquery-1.7.2.js"></script>
  <meta charset="utf-8">
  <title>JS Bin</title>
<style id="jsbin-css">
.hallucination {background-color:rgb(208, 116, 116)}
.orange {background-color:rgb(201, 165, 99)}
.green {background-color:rgb(115, 167, 115)}
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

    if (Math.abs(score) >= user_threshold) {
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
  
<form>
  <label for="user_threshold">Threshold:</label>
  <input type="text" id="user_threshold" name="user_threshold" value="0.9">
  <button type="button" id="submit_button" onclick="display_highlights()">Submit</button>
</form>
"""

html_bottom = """
    </body>
    </html>
"""


def iter_lines(infile):
    with open(infile, 'r',encoding='utf8') as inf:
        for line in inf:
            yield json.loads(line.strip())

def get_simple_html(data):
    tokens = ['<div>']
    for score, token in zip(data['diff_pos_scores'], data['target_tokens'].split()):
        html = f'<span score={score}>{token.replace("▁", " ")}</span>'
        tokens.append(html)
    tokens.append('</div>')
    return ''.join(tokens)

if __name__ == '__main__':
    infile = sys.argv[1]
    print(html_top)
    for data in iter_lines(infile):
        print(get_simple_html(data))
    print(html_bottom)