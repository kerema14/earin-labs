## Online Usage with SWISH

You can run the `variant4_to_num.pl` code directly in your browser using SWISH:

1. Open your browser and go to:  
   https://swish.swi-prolog.org/

2. In the top pane (the editor), paste the entire contents of `variant4_to_num.pl`.

3. Click the **“Save”** button (you may give it any name, e.g. `to_num.pl`).

4. In the bottom pane (the query window), enter your queries. Press **Run** or hit **Ctrl + Enter** to execute.

### Example Queries

```prolog
?- to_num("ninety nine", N).
% Expected result: N = 99.

?- to_num("one hundred and one", N).
% Expected result: N = 101.

?- to_num("seven hundred and eighty six", N).
% Expected result: N = 786.

?- to_num("one thousand", N).
% Expected result: N = 1000.
