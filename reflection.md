# Reflection on Assignment 5

* **Answer the first Section (e) question.**

Larger windows give word vectors that correspond better to semantic similarity, while small windows mainly capture syntactic (and some level of semantic) similarity.

* **Answer the second Section (e) question.**

Here are some that I tried:

expensive - nice + awful = overpriced (good)

pasta - italian + chinese = noodles (good)

sunny - summer + winter = weather (nope)

wine - italian + american = beer (good)

latte - coffee + tea = milk (meh)

* **Answer the third Section (e) question.**

(Examples) Word lengths: one dimension per integer length, where the value is the number of words in the review with that length. Word bigrams (to get simple phrases): one dimension per bigram, where the value is the count of that bigram in the view.

* **Answer the fourth Section (e) question.**

Many of the mistakes resulted from the reviewer contrasting the place to something else, and/or negations. For example, phrases like "barely resisting the great smashburger right next door" and "the dining room comfort was ok , not great" in a poor review contain terms that are more indicative of positive reviews. Conversely, "yes , i know it sounds strange and not necessarily appealing" in a good review fool the classifier as well. 

Bigrams would help to catch the negations ("not great"), but we'd require more sophisticated discourse-level features to identify whether a sentence is referring to the establishment in question or something else.
