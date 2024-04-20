# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

args = sys.argv
model = args[1]

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")


input_file_path = "./raw_data.txt"

raw_generation_file_path = "./raw_generation_output_ten-shot.txt"

output_file_path = "./raw_data_output_ten-shot.txt"

with open(input_file_path,"r",encoding="utf-8") as file:
    input_data = file.readlines()
input_data = input_data[:5]

requirement_prompt = """Assuming you are a software requirements analyst 
you are categorizing user comments into three categories (bug report, feature request, and irrelevant information).
bug report refers to the failure of the original software functionality;
feature request is a structured request applied to software maintenance that implements a new function or enhance existing features;
irrelevant information refers to information that does not contain information that can be applied to software updates.
"great communication ap for our basketball team. i wish the calendar was more of a ""monthly view"" versus just the event day...great ap though!"->feature request
Cool app but if i log out once and i know the password and try to login many times it always say pass incorrect or some other issues so i make other account and if log out for some personal reasons nd agn try to login it again do that thing i hate this pls it is just me or anyone else discord please make my account come back TvT..->bug report
There is no way to jump to first post. Newest release break my zoom settings/font size->feature request
2 days ago this would have been a 4 star review but the new update is such a downgrade that I had to say something. Why they always insist on making every feature worse is beyond me->irrelevant information
So far, all is good. We use this handy app with our church community.->irrelevant information
This a good app but there needs to be an edit button option.->feature request
I wish there was a way to send a reminder directly from a calendar event to anyone who hasn't responded. Otherwise, it's great!->feature request
It has been good.->irrelevant information
The website is great, I got the app on my chromebook so I could send voice messages, but it doesn't let me record anything.->feature request
show any error message but ask each time for password->bug report
Allows me to stay in touch easily and get information to my teams and activity groups.->irrelevant information
"new ui sucks, cant revert to old one ""we changed it for mobile instead of a scaled down desktop version"" thats literally why we liked the mobile app because it didnt try to feel like a f**ckin mobile app. you cant see full pinned messages or their attachments and the new server menu requires endless scrolling for what used to be a seamless experience with desktop. discord recently has been fixing things that werent broken in the first place and completely ignoring community feedback. why."->feature request
Recent updates is bugging my app. It keeps on lagging (the screen showing one channel despite switching to other channels in the server).->bug report
My screen keeps turning black->bug report
Does what it's supposed to do->irrelevant information
I enjoy the chats..->irrelevant information
It's a great way to communicate coworkers. And very easy to use->irrelevant information
sometimes it works and other times it doesn't; will drop calls periodically->bug report
App is extremely buggy. Sometimes it won't load a group, and notifications don't come half the time->bug report
Love the app. It helps me keep up with all of my boys football schedules while getting a shingle full time working mom who is always busy. Makes life easier->irrelevant information
Worst review we cant hear each ither can't share the screen and the skype is not working well and I didn't got ant response from team skype how you are all working->bug report
How do you not have the option to delete chats, or even check who has seen your message? Garbage. Do yourself a favor and get WhatsApp. The functionality of this app is atrocious.->feature request
Wont even make video/voice calls, it just stops ringing. Works 50% of the time.->bug report
This is a great way to communicate with others. I would recommend this to anyone.->irrelevant information
It's slow to load->bug report
Discord is already part of my day-to-day routine. But when it was updated, and I'm on call and trying to open other apps to play while on a VC, I'm automatically disconnected from the call and can't join the VCs unless I'm on the app itself. That's why I can't join calls when trying to play with my friends or I can play but can't join the VC.->bug report
I've used Discord for 7 years and I've gotten so used to this mobile layout. My app finally automatically updated today, against my own wishes due to what I've heard about the new UI, and I've gotta say I'm confused and disappointed. What happened to swiping left to check the member list? I've gotta tap on the channel name now to check that? Never mind the fact that I can't find any semblance of a friends list anymore, and all of my direct messages are in that new bottom toolbar, unlike on PC.->feature request
It's good but please upgrade it so that it can be used without network->feature request
Easiest way to talk with people online atm.->irrelevant information
it would be a 4 but the update is absolutely horrible. the new layout makes it look so cluttered, and the new call background is just why. it's so bad I can't even.->feature request"""

with open(output_file_path,"w",encoding='utf-8') as output_file,\
        open(raw_generation_file_path,"w",encoding='utf-8') as raw_generation_output_file:
    for text in input_data:
        content = text.strip()
        input_text = requirement_prompt +"\n"+ content.strip() + "->"
        input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = model.generate(**input_ids, max_new_tokens=10)
        
        raw_generation = tokenizer.decode(outputs[0]).split("->")[-1]
        result = raw_generation.split("\n")[0]
        output_line = f"{content}\t{result}\n"
        output_file.write(output_line)
        raw_generation_output_file.write(raw_generation + "\n")
        print(result)

print(" annotate over ! ")
