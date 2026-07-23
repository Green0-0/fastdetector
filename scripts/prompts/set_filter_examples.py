from fastdetector.prompting.prompts import load_prompts
from fastdetector.prompting.prompt_builder import add_example, save_dataset

prompts_path = "prompts/filter_contiguous_subset.json"

prompts = load_prompts([prompts_path])

for p in prompts:
    p.examples = []

input_format = "{{DOC}}\nExtract the longest contiguous, logical subset of the document that contains no boilerplate or metadata (headers, dates, bylines, navigation, image.png, READ MORE, and similar noise). You should not have any removed boilerplate inbetween the text, the new text you extract must be a strict contiguous subset of the original text. Keep the original wording and order: every newline, space, punctuation should be copied exactly as is with no deviations. Return only the extracted text with no extra commentary. If no valid subset exists, return NONE."

doc_1 = """
Firearms Training Articles
Firearms Training Articles
Firearms Retailers & Gun Stores in BC
Firearms Retailers & Gun Stores in BC
Gun Ranges & Clubs in BC
Gun Ranges & Clubs in BC
Latest Articles
The Silvercore Podcast Ep. 10 – Moose Underwear and Other Hunting Stories       
Pro Tips for Renewing your Firearms License     
The Silvercore Podcast Ep. 9 – Grizzly Bear Attack      
The Silvercore Podcast Ep. 8 – Train Your Brain To Win Part 2   
The Silvercore Podcast Ep. 7 – Train Your Brain To Win  
Silvercore has been providing bear safety training to industry professionals, both in person as well as online, for many years, but this marks the first time we speak with someone who survived a horrific grizzly bear attack by using his pocket knife.
In episode 9, we sit down and talk with Colin Dowler, who shares in detail his harrowing experience of grit and tenacity. Colin is sharing his story to increase the knowledge base of bear behaviour and bear encounters.
READ MORE
He hopes that we all become more aware of potential dangers and become adequately educated and prepared before entering into bear country.
Please note that sections of this episode deal with intense and graphic content, listener discretion is advised.
Watch the Podcast on YouTube here: https://www.youtube.com/watch?v=BErmHQkeEAk
You can listen to episode 9 of our podcast on Podbean, Apple iTunes, YouTube, Spotify, SoundCloud, Google Podcast, and Google Play. All you'll have to do is search for 'The Silvercore Podcast.'
If you have any feedback or questions that we can address, please reach out to us via social media or at 1-855-771-5837 or This email address is being protected from spambots. You need JavaScript enabled to view it.. Finally, don't forget to rate, review, and subscribe to the podcast, and while you’re at it, follow us on Facebook, Instagram, and Twitter!
Travis Bader
Silvercore Inc.
Keep Up With Useful and Fun Firearms Training On Social Media
Free Firearms Safety
""".strip()
doc_1_filtered = """
Silvercore has been providing bear safety training to industry professionals, both in person as well as online, for many years, but this marks the first time we speak with someone who survived a horrific grizzly bear attack by using his pocket knife.
In episode 9, we sit down and talk with Colin Dowler, who shares in detail his harrowing experience of grit and tenacity. Colin is sharing his story to increase the knowledge base of bear behaviour and bear encounters. 
""".strip()
doc_2 = """
1600t mill per hour     
Home
Products
Solutions
Project
About
Contact
39
C
1600t mill per hour
800t suspension roller mill per hour
600 supplier 80 tons per hour mobile crusher greenrevolution jaw crusher for sale jaw crusher design price 5 ton per hour mobile gold process mill equipmentplantproject our pany has jaw 30th read more jaw crusher structure in niger Chat With Sales ...
Get Price
100 ton per hour gold mill
100 Ton Per Hour Gold Mill 10,000 TPD & 3,000 TPD Gold Processing Plants, 900 TPH . 10,000 TPH Gold Processing Mill Plant - Edmonton & Calgary, Canada Description: - 4' Standard Cone Crusher with 100 HP motor. Get Information; 50 ton per hour gold
Get Price
1000t airflow mill per hour
1000T universal mill per hour 1000t airflow mill per hourYouTube. Nov 21 2019 This video is unavaile. Watch Queue Queue. Watch Queue Queue. Chat Now. how to measure revolution per minute of a ball millusing a . how to measure revolution per minute of
Get Price
100 ton per hour ball mill in lebanon
100 ton per hour ball mill in lebanon 10 100 Tons Per Hour Ball Mill Price,You are Here Home 10 100 Tons Per Hour Ball Mill Price Cost on Setup Palm Oil Processing Mill in Nigeria A complete palm oil processing mill plant with a capacity of 50 tonday is about ...
Get Price
100 tons per hour rice mill complete set project
To illustrate the rice business, A small processor with a 1 ton per hour mill, Using the farmers 3 hectare farm as base where he harvest 4 tons per hectare, Read More Grain Processing Technology - CIMBRIA 40 tonnes per hour. Steam heated. capacity from ...
Get Price
""".strip()
doc_2_filtered = """
NONE
""".strip()
doc_3 = """
John Wesley's Bible Notes and Commentary - Genesis 16
Bad Advertisement?
Are you a Christian?
Online Store:
Visit Our Store
JOHN WESLEY'S BIBLE COMMENTARY
NOTES - GENESIS 16
Genesis 15 - Genesis 17 >> - HELP - FB - TWITTER - GR VIDEOS - GR FORUMS - GR YOUTUBE
XVI Hagar probably was one of those maid - servants which the king of Egypt (among other gifts) bestowed upon Abram, chap. xii. 16. Concerning her we have four things in this chapter,
I. Her marriage to Abram her master, ver. 1-3.
II. Her misbehaviour towards Sarai her mistress, ver. 4-6.
III. Her discourse with an angel that met her in her flight, ver. 7- 14.
IV. Her delivery of a son, ver. 15, 16.
Verse 1. We have here the marriage of Abram to Hagar, who was his secondary wife. Herein, though he may be excused, he cannot be justified; for from the beginning it was not so: and when it was so, it seems to have proceeded from an irregular desire to build up their families, for the speedier peopling of the world. But now we must not do so? Christ has reduced this matter to the first institution, and makes the marriage union to be between one man and one woman only.
Verse 4. We have here the ill consequences of Abram's marriage to Hagar: a deal of mischief it made presently. Hagar no sooner perceives herself with child, but she looks scornfully upon her mistress; upbraids her perhaps with her barrenness, and insults over her. Sarai falls upon Abram, and very unjustly charges him with the injury, suspecting that he countenanced Hagar's insolence: and as one not willing to hear what Abram had to say she rashly appeals to God. The Lord judge between me and thee, as if Abram had refused to right her. When passion is upon the throne, reason is out of doors, and is neither heard nor spoken. Those are not always in the right that are most forward in appealing to God. Rash and bold imprecations are commonly evidences of guilt and a bad cause.
READ MORE
""".strip()
doc_3_filtered = """
XVI Hagar probably was one of those maid - servants which the king of Egypt (among other gifts) bestowed upon Abram, chap. xii. 16. Concerning her we have four things in this chapter,
I. Her marriage to Abram her master, ver. 1-3.
II. Her misbehaviour towards Sarai her mistress, ver. 4-6.
III. Her discourse with an angel that met her in her flight, ver. 7- 14.
IV. Her delivery of a son, ver. 15, 16.
Verse 1. We have here the marriage of Abram to Hagar, who was his secondary wife. Herein, though he may be excused, he cannot be justified; for from the beginning it was not so: and when it was so, it seems to have proceeded from an irregular desire to build up their families, for the speedier peopling of the world. But now we must not do so? Christ has reduced this matter to the first institution, and makes the marriage union to be between one man and one woman only.
Verse 4. We have here the ill consequences of Abram's marriage to Hagar: a deal of mischief it made presently. Hagar no sooner perceives herself with child, but she looks scornfully upon her mistress; upbraids her perhaps with her barrenness, and insults over her. Sarai falls upon Abram, and very unjustly charges him with the injury, suspecting that he countenanced Hagar's insolence: and as one not willing to hear what Abram had to say she rashly appeals to God. The Lord judge between me and thee, as if Abram had refused to right her. When passion is upon the throne, reason is out of doors, and is neither heard nor spoken. Those are not always in the right that are most forward in appealing to God. Rash and bold imprecations are commonly evidences of guilt and a bad cause.
""".strip()

add_example(prompts, (input_format.replace("{{DOC}}", doc_1), doc_1_filtered))
add_example(prompts, (input_format.replace("{{DOC}}", doc_2), doc_2_filtered))
add_example(prompts, (input_format.replace("{{DOC}}", doc_3), doc_3_filtered))

save_dataset(prompts, "filter_contiguous_subset.json", path="prompts/")
print(f"Added {len(prompts[0].examples)} examples to {prompts_path}")