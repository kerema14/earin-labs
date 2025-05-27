% Prolog implementation for Variant 4: Convert English words (up to one thousand) into numerical digits

% Entry point: to_num(+Text:string, -Number:int)
% Example usage:
% ?- to_num("ninety nine", N).
% N = 99.
% ?- to_num("one hundred and one", N).
% N = 101.
% ?- to_num("one thousand", N).
% N = 1000.

to_num(Text, N) :-
    % Split the input string into words
    split_string(Text, " ", "", StrWords),
    % Normalize to lowercase
    maplist(string_lower, StrWords, LowerStrs),
    % Convert each substring to an atom
    maplist(atom_string, Words, LowerStrs),
    % Parse the list of word-atoms into a number
    phrase(number(N), Words),
    !.

% Grammar for numbers up to 1000
number(N)   --> thousand(N).
thousand(1000) --> [one, thousand].
thousand(N) --> hundreds(N).

% Handle 0..999
hundreds(N) --> hundreds_part(H), remainder(R), {N is H + R}.
hundreds_part(H) --> units(U), [hundred], {H is U * 100}.
hundreds_part(0) --> [].

% Optional 'and' and tens/units
remainder(R) --> [and], tens_units(R).
remainder(R) --> tens_units(R).
remainder(0) --> [].

% Tens and units (0..99)
tens_units(N) --> tens(T), units_digit(U), {N is T + U}.
tens_units(N) --> tens(N).
tens_units(N) --> units(N).

% Mapping for units 0..19
units(N) --> [Word], {unit_map(Word, N)}.

% Units digit for combining (1..9)
units_digit(U) --> [Word], {unit_map(Word,U), U > 0, U < 10}.
units_digit(0) --> []. % for the cases like eleven, twelve, thirteen...

% Mapping for tens 20,30,...,90
tens(N) --> [Word], {tens_map(Word, N)}.

% Fact tables
unit_map(zero,  0).
unit_map(one,   1).
unit_map(two,   2).
unit_map(three, 3).
unit_map(four,  4).
unit_map(five,  5).
unit_map(six,   6).
unit_map(seven, 7).
unit_map(eight, 8).
unit_map(nine,  9).
unit_map(ten,   10).
unit_map(eleven,    11).
unit_map(twelve,    12).
unit_map(thirteen,  13).
unit_map(fourteen,  14).
unit_map(fifteen,   15).
unit_map(sixteen,   16).
unit_map(seventeen, 17).
unit_map(eighteen,  18).
unit_map(nineteen,  19).

tens_map(twenty,  20).
tens_map(thirty,  30).
tens_map(forty,   40).
tens_map(fifty,   50).
tens_map(sixty,   60).
tens_map(seventy, 70).
tens_map(eighty,  80).
tens_map(ninety,  90).