---
title: The 'Backyard' Chronicles
date: 2024-09-20 07:00:00 +1000
categories: [Backyard]
tags: [machine-learning]
desicrption: An in-depth dive into the technical details and development journey
---
The *'Backyard'* is a culmination of over a years worth of work. Looking back, it's incredible how far it has come from a vague idea in my mind to a tangible product. I have had to learn a lot and persevere through as I entered somewhat uncharted territory. The *'Backyard'*, as far as I'm aware, is truly one-of-a-kind and, although not perfect, I'm incredibly proud of what it is. I want to take a deep-dive into the journey of how it was made, the technical triumphs and failures encountered and reflect back on why I decided to make it.

## What is the 'Backyard'?
In Guilty Gear lore, the *'Backyard'* is described as:
> ...kind of a metaphysical command prompt. It's a realm of pure information, and all of that information makes up our collective reality - [TheGamer](https://www.thegamer.com/guilty-gear-the-backyard-explained/)

However Guilty Gear lore is a far more complicated than this entire project and is a rabbithole only delved into by the most dedicated or deranged of Guilty Gear enthusiast. As the sum of all information, I thought the *Backyard* was an appropriate name to give to a suite of tools that visually analyse matches from *Guilty Gear -Strive-* and attempt to further understand the factors involved in winning. If you have not yet, I would suggest going to [backyard-insight.info](https://backyard-insight.info/) to see the final result. For a full explanation of each tool, I suggest going to the appropriate GitHub of *[Backyard-Observer](https://github.com/tmltsang/Backyard-Observer)* and *[Backyard-Insight](https://github.com/tmltsang/Backyard-Insight)* but I'll give a brief summary of each tool below.

### Backyard-Observer
*[Backyard-Observer](https://github.com/tmltsang/Backyard-Observer)* is mainly reponsible for training and utilising [YOLOv8](https://docs.ultralytics.com/) vision models for metrics gathering.

![Metrics Gathering](https://media.githubusercontent.com/media/tmltsang/Backyard-Observer/main/img/docs/backyard-observer.gif)
_Example of metrics gathering_

It is also capable of 'Real-Time' predictions, although this feature is still a proof-of-conecpt.

![Predictions](https://media.githubusercontent.com/media/tmltsang/Backyard-Observer/main/img/docs/prediction.gif)
_'Real-Time' predictions'_

Check out the [README.md](https://github.com/tmltsang/Backyard-Observer/README.md) if you'd like to know more.

### Backyard-Insight
A dashboard hosted at [https://backyard-insight.info/](https://backyard-insight.info/). It's main feature is that it predicts tournament match results based on any given match state. The dashboard also features some counting stats such as number of bursts and tensions bars used in a match.

![Backyard-Insight](https://cdn.jsdelivr.net/gh/tmltsang/Backyard-Insight/docs/images/demo.gif)
_Dashboard of Nitro vs Tatuma at EVO Grand Finals_

### Notebooks
* [ggstrive.ipynb](https://colab.research.google.com/drive/1ybJt9Y1jr8Qtdvq8T515--zxLptH8D7v?usp=sharing) - Data exploration and machine learning model training for the Win Predictor.
* [ggstrive_asuka.ipynb](https://colab.research.google.com/drive/1HPtgk7gfxv6YQVEiv5CYf8RlGwLRczoV?usp=sharing) - Data exploration and machine learning model training for the Asuka Spell model.
* [ggstrive-Tournament.ipynb](https://colab.research.google.com/drive/1_gkzzw3t4O7hxUaud6jyS6_gkZBsgGU-?usp=sharing) - Data cleaning for tournament data.


## Origin
The idea was conceieved, as all good ideas are, in frustration as I bounced back and forth between Floor 8 and Floor 9 of the 'Tower', the rank system of *Guilty Gear -Strive-*. As I struggled to climb the floors, I brainstormed ways I could improve as I scrolled past the 'Training Room' and 'Replay Centre'. Could I make a tool that compared my gameplay against that of top players and determine what actions influence winning? Maybe I could try machine learning, there are a multitude of library and resources. How hard could it be?[^footnote]

In all serious, I was also inspired by the sheer amount stats traditional sports have and how that improves the viewing experience. [DeepStrike](https://www.jabbr.ai/deepstrike) was just starting to gain recoginition and its ability to track and represent the data autonomously using computer vision for combat sports, the closest traditional sport parallel to fighting games, was fascinating. At this time I didn't have a lot of expertise in the area but I wanted to see if I could provide a similar experience for e-sports.

### Investigation and Exploration
Initially I wanted to track every single action a character performed in match —movement, attacking, blocking, etc...— and see the difference in actions taken by low and high ranked players. In my case, that character would be my main, Testament. To help me get started with the computer vision, I first followed this incredibly helpful tutorial by [Learn Code by Gaming](https://learncodebygaming.com/blog/tutorial/opencv-object-detection-in-games).

It used **template matching** which the OpenCV documentation describes, *"It simply slides the template image over the input image (as in 2D convolution) and compares the template and patch of input image under the template image"*[^footnote-2]. Adapting this to my use case, is it possible to take the images of every move from [dustloop](https://www.dustloop.com/w/GGST) (the wiki of *Guilty Gear -Strive-*) as a template image and accurately determine if the move was used in a frame of a video. To start off with I would try with **Testament's far slash**.

![Testament's Far Slash](https://www.dustloop.com/wiki/images/3/38/GGST_Testament_fS.png){: .w-50}
_Testament's far slash from dustloop_

![Testament template matching](../assets/img/template_matching.png)
_Testament's far slash found within frame of a video with template matching_

Success! A green rectangle is drawn around a match with confidence over **0.85**. In the example above the output was `Best match confidence: 0.8679274916648865`. Let's try with a video now.

![Testament template matching](../assets/img/template_over_matching.gif)
_Many false positives..._

That's not good. It is completely unreliable. To make it robust, I tried to use *feature detection and matching*. To give a simple overview of how this works, a set of *features* are determined on the template image. Usually this entails edges or corners anything distinct points that can be used. The same *feature detection* is then used on the larger input image and an attempt is made to match similar features. (The OpenCV documentation has a much more in-depth explanation if you are interested)[^footnote-3]. However this turned out to also be ineffective.

![Testament template matching](../assets/img/sift_orb_matching.png)
_Feature detection and matching with ORB_

It may have been possible to tweak some parameters to make this work more consistently however there was another issue.Each move needs to also be flipped as characters can face both directions, when the flipped image was added for detection the performance greatly suffered. That was only with 2 images; Each character has 5 attacks in standing, crouching and jumping position, command normals, special moves, overdrive. That's not even considering the fact that attacks have multiple frames. It was clear that this solution just would not scale and efforts were better placed elsewhere.

## When in Doubt - YOLO
As I was doing some research I stumbled upon this video:

{% include embed/youtube.html id='LXA7zXVz8A4' %}
_Credit: River's Educational Channel_

 I suggest watching it all if you have some time. It's a really fascinating video where they try to create a bot in Valorant using computer vision. They end up using YOLO for their model for its quick real-time detection. It wasn't perfect however the issues they were facing may not necessarily apply to me. The issue with Valorant, a 3-D tactical shooter, is that you can objects can be viewed from an angle or distance. In a 2-D fighters such as Guilty Gear, the camera is at a static angle and distance, only panning left and right. YOLO seems like it could be an ideal choice for this project. There is no existing dataset for *Guilty Gear -Strive-*, so I'll need to annotate it myself using [labelimg](https://github.com/HumanSignal/labelImg).

### My Nightmare - Data Annotation

 ![Testament annotation](../assets/img/labelimg_testament.png)
 _My life for weeks..._

I wanted to track a few key objects and actions:

Testament
: All actions associated with Testament, including movement, attacks and blocking

Objects associated with Testament
: This includes the crow, stain state, succubus and the skull projectile from Grave Reaper

Opponent
: Differentiating between Testament and any other character and mainly seeing if they are blocked or were hit by an attack

System Mechanics
: All Bursts and Roman Cancels

This ends up being over 60+ different classes to annotate and track. After a week I was able to annotate around 200+ images and start training the vision model. However results early were not very promising.

 ![Testament initial try](../assets/img/testament_initial_try.gif)
_At least it tracks the crow..._

It's very quick but it's struggling to really identify Testament at all. Time to annotate more data...

In retrospect I have regrets about how I handled this. I didn't do a lot of planning, I was just enjoying learning about new technologies and anything more than just exploring I thought would have ruined it for me. That being said without a clear, achievable goal data annotation is truly a tedious, menial task that drained my optimism about this project. Without a vision of what I wanted to achieve I thought my only way forward was to perfect the model. This just created of negative feedback loop where I would begrudingly annotate hundreds of images, train the model, be frustrated by any flaws I could see and go back to annotating. I knew vaguely that the project consisted of training a vision model -> metric gathering -> visualisation. So potentially I could have picked a move that I knew the model was interpreting correctly , such as Testament's far slash, and create an intermediary milestone to build the end-to-end system that tracks how many times it is used and if it was blocked or not. It would have given me something realistic to aim for and it would allow me to not be stuck only with annotation work.

The lack of planning also hurt as if you don't plan out all the classs you want to track, adding them in later can be painful. Multiple times I had to go back over hundreds of already annotated images, to annotate them again because there was a new class that needed to be added.

Besides annotation, I tweaked the training models and parameters. The main breakthrough was discovering the different [model sizes](https://docs.ultralytics.com/tasks/detect/#models). Up until this point I had been using the smallest size, YOLOv8n which while quick, was only able to detect a few classes. Increasing this to the largest class, YOLOv8x, improved the quality greatly.

![Testament final model](../assets/img/testament_final_vision_model.gif)
_Tracks most of Testament's actions correctly_

This does comes with some consequences, there is an impact on performance in both training and running. While this is a lot better it's not quite accurate enough. I could keep annotating more images however I was starting an issue.

![Testament final model](../assets/img/testament_labels.jpg)
_Class Distribution of annotated images_

The top left of the graph shows the distribution of classes. As you can see there is a huge imbalance of examples of each class. This is expected to a degree, the two classes with huge spikes are 'Testament's Crow' and the 'Opponent' which are in almost every example but there are moves that rarely turn up. I've been obtaining frames by splitting up videos of high-level Testament matches from youtube. This means I don't get an equal distribution of moves as obviously players tend to gravitate to better moves.

Regardless with this model, I attempted to make some visualisations. If I could track every single move a character does throughout the match it could then be represented as a **directed network graph**.

![Testament move chart](../assets/img/testament_move_chart.png)
_I did say 'attempt'_

I was not happy with the model and the visualisation I wanted to do is a complete mess. I could keep annotating images to hope I improve the accuracy of the model I was satisfied with however at this point I realised something. Earlier on I had stated as my initial goal:
> Could I make a tool that compared my gameplay against that of top players and determine what actions influence winning?

Yet I had completely ignored the most fundamental part, who wins a match?

## Starting Over
This is the beginning of *Backyard Insight* and *Observer* as it is now. Instead of looking at the actions of a character on the screen the values of the different **gauges** and **system texts** could be tracked and then into a machine learning model and predict the percentage probabilty that a player could win in the current state. The effects of different gauges on winning could then be examined. How important is **Tension** and **Burst** and different health values? The eventual goal would be to use this model to analyse tournament matches and provide a prediction throught the match in manner similar to chess, where modern chess engines inspeect the strength of the positions of each player.

![chess](../assets/img/chess_engine.png) {: .w-50}
_Bar on the left showing the strength of player positions_

 To achieve this the following steps would need to be completed:
1. Annotate and create a dataset for a vision model
2. Train the vision model
3. Use the vision model to create a dataset for a predictive machine learning model
4. Train the predicitive model
5. Run the vision model on tournament matches to create a tournament dataset
6. Use the predictive model on the tournament dataset
7. Create a dashboard from the prediction and tournament dataset

## Determining Classes to Track
Before starting classes for the vision model needs to be determined. It wasn't clear exactly what would be needed down the line so as much as possible was tracked. This includes:

![ggstrive glossary](../assets/img/gg_strive_glossary_example.jpg)


[Health](https://www.dustloop.com/w/GGST/Damage#Guts_and_Defense_Ratings)
: Also known as Life Gauge, the amount of remaining health that a character has, once this reaches 0 the character loses the round

Damaged
: If a player is currently being damaged, the healthbar will have a red segment on it.

[Burst](https://www.dustloop.com/w/GGST/Mechanics#Burst_Gauge)
: Used primarily for Psych Burst, a defensive option that consumes 100% Burst to interrupt and knock back an opponent. Starts at 100% at the beginning of a match and is the only gauge to be carried over between rounds. Season 2 added 'Wild Assault' and 'Deflect Shield' which both consume 50% Burst

[R.I.S.C.](https://www.dustloop.com/w/GGST/Damage#R.I.S.C._Level)
: A gauge that builds as a player blocks attacks. It depletes when a player is hit and they will take additional damage depending on how full the bar was

[Tension](https://www.dustloop.com/w/GGST/Mechanics#Tension_Gauge)
: The main 'resource' in Guilty Gear. Can be spent on a variety of offensive and defensive options. Most actions consume 50% tension which is marked by a 'gear' icon on the bar.

Round Count
: Represented by the hearts above the healthbar, become grey and broken when a round is lost

![ggstrive_system_text](../assets/img/ggstrive_system_text.png)
_Source: Dustloop[^footnote-4]_

[Counter](https://www.dustloop.com/w/GGST/Mechanics#Counter_Hit)
: Hitting an opponent before their recovery results in a powerful 'Counter' hit state

Reversal
: Performing an action immediately as they recover. Can be potent if combined with invincible moves

[Just/IB](https://www.dustloop.com/w/GGST/Mechanics#Instant_Block)
: IB stands for Instant Block. If a player blocks within 2 frames of their opponents attack, large 'JUST' text appears on screen and they are benefitted with an advantageous state

Punish
: If a player attacks their opponent during their recovery, the word 'Punish' will flash on their screen

![ggstrive_system_text](../assets/img/ggstrive_round_start.jpg){: .w-50}

Round Start
: Each round start with the same **Let's Rock** graphic.

![ggstrive_system_text](../assets/img/ggstrive_slash.jpg){: .w-50}

Round End:
: Both 'Slash' and 'Perfect' screens can indicate the end of a round. Additionaly the **Player 1** and **Player 2** text is also tracked


## My Recurring Nightmare - Data Annotation
 I knew this would involve in making another YOLOv8 vision model for this but it should be easier this time. The gauges are UI elements and should be on screen at most times and this makes class distribution consistent. Each **gauge** is distinct in colour, location, appearance and take a signifianct portion of the screen. This should result in less annotated frames needed overall. Although the **system texts** are more situational, there is still an abundance of examples for most of them in a typical match.

![ggstrive_system_text](../assets/img/bar_labels.jpg)
_Class distribution from initial training_

The class distribution is much better resulting in good results early on. After the initial training I implemented an iterative process where I would, train a vision model -> Use the model to create labels on new frames -> Adjust any incorrect labels -> train the vision model with the new frames. This allowed me to quickly annotate frames at a much higher rate.

![ggstrive_bar_confusion_matrix](../assets/img/bar_final_confusion_matrix.png)
_Confusion Matrix during a later training_

The confusion matrix shows the actual class against the predicted class. An ideal matrix would only have points along the diagonal as that is where the actual class and predicted class intersect. In this model there are a few minor errors however they are primarily 'background', which is used when no prediction is made. This is not a big deal as the videos run at 60fps (frames per second) and missing the value of a gauge on a singular frame is not the end of the world.

## At Last... Data Collection
The next task was to use this vision model to record a match.
### Considerations
Data Format:
: The data format would need to be able represent every variable over the course of a match. This could be represented as a table, the columns are each variable and the rows are a specific point in time and saved to a CSV file.

Variables:
: There are two type of variables that need to be determined before training a predictive machine learning model, **features** or **input variables** and **target** or **output variables**. The **features** are variables that will predict or explain the **target**, otherwise known as the outcome. The machine learning model will try to find a relationship between the different features that best determines the target. In this case, features are relatively straight forward it's mainly composed of the value of each *gauge* repeated for both players and the number of times an action with system text occurs e.g. counter hit, reversal...

#### Features
| Column Name     | Type      | Description                                                 |
| --------------- | --------- | ----------------------------------------------------------- |
| time            | `float`   | Time elapsed in current round                               |
| p1_name         | `string`  | Name of character in P1 position                            |
| p2_name         | `string`  | Name of character in P2 position                            |
| p1_health       | `float`   | Percentage of health remaining for P1 between 0-1           |
| p2_health       | `float`   | Percentage of health remaining for P2 between 0-1           |
| p1_tension      | `float`   | Percentage of tension filled for P1 between 0-1             |
| p2_tension      | `float`   | Percentage of tension filled for P2 between 0-1             |
| p1_burst        | `float`   | Percentage of burst filled for P1 between 0-1               |
| p2_burst        | `float`   | Percentage of burst filled for P2 between 0-1               |
| p1_risc         | `float`   | Percentage of burst filled for P1 between 0-1               |
| p2_risc         | `float`   | Percentage of burst filled for P2 between 0-1               |
| p1_round_count  | `int`     | Number of rounds won by P1 i.e. number of hearts lost by P2 |
| p2_round_count  | `int`     | Number of rounds won by P2 i.e. number of hearts lost by P1 |
| p1_counter      | `int`     | Number of 'Counters' performed by P1                        |
| p2_counter      | `int`     | Number of 'Counters' performed by P2                        |
| p1_just         | `int`     | Number of 'Just'/IBs performed by P1                        |
| p2_just         | `int`     | Number of 'Just'/IBs performed by P2                        |
| p1_punish       | `int`     | Number of Punishes performed by P1                          |
| p2_punish       | `int`     | Number of Punishes performed by P2                          |
| p1_reversal     | `int`     | Number of Reversals performed by P1                         |
| p2_reversal     | `int`     | Number of Reversals performed by P2                         |
| p1_curr_damaged | `boolean` | If P1 is currently being damaged                            |
| p2_curr_damaged | `boolean` | If P2 is currently being damaged                            |

#### Targets
However the standard format for *Guilty Gear -Strive-* (and most other fighting games) is played over a best of 3 'rounds' known as set/match. While most gauges reset between rounds critically 'Burst' is actually carried over throughout the set i.e. if used in round 1 it may not be availabe for round 2. This neccessitates 2 **target** variables, *Round Win* and *Set Win* and therefore most likely 2 machine learning models.

| Column Name  | Type      | Description               |
| ------------ | --------- | ------------------------- |
| p1_round_win | `boolean` | If P1 wins this round     |
| p1_set_win   | `boolean` | If P1 wins this match/set |


### Recording Data
To finally begin recording data the YOLOv8 output needs to be interpreted to the CSV data format detailed above. This doesn't account for the character names for `p1_name` and `p2_name` as that is determined from the filename. The pseduo-code looks for the logic looks like this:
```python
read YOLOv8 frame
if 'round_start' in frame:
  wait until 'round_start' disappers # This will be the first fame of the actual round
  update 'round_count' # Look at the number of hearts vs hearts broken in frame for each player
  initialise new 'round'
# Slash is the round end graphic time to determine winner
elif 'slash' in frame:
  #Determine winner from health values
  if p1_prev_health > p2_prev_health:
        winner = P1
  else:
        winner = P2
  round_history['round_winner'] = winner
  set_history.append(round_history)
  # round_count + 1 will be equivalent to number of rounds won at the end of a round
  if 'round_count' + 1 == 'max_round_count':
    set_history['set_winner'] = winner
    record_set_to_csv(set_history)
else:
  for each 'feature':
    if 'feature' in frame:
      current_round['feature'] = current frame value
    else:
      current_round['feature'] = previous frame value
  round_history.append(current_round)
```
{: file='bar_collection.py'}

To test, this was successfully run against a handful of videos. The full data collection can now begin. For this the videos used were sourced from [GGST: High Level Gameplay](https://www.youtube.com/@GGHighLevel) totalling to about ~238 videos with a roughly equal character distribution. The videos were processed in alphabetical order starting with 'aba', I went to bed hoping that it would be done once I worke up. Checking in the morning, it had made it to 'asuka'.... It was still on 'a', this might take a while.

I had severley underestimated how long this may take. The average video length was ~10 minute and with 238 videos that equates to around 40hrs of videos to process. However the data collection runs slower than 'real-time' i.e. 60 frames-per-second. The main bottleneck was the GPU, a 2080 with only 8GB of VRAM. An easy way would be to skip frames as it's not necessary to process every frame but what would be the appropirate amount of frames to skip? Fighting games rounds are not long, most not lasting longer than a minute and therefore losing too much fidelity would be signifcantly detrimental to accuracy. In the end, every 6th frame, every 0.1 seconds, seemed to be the right balance of speed and fidelity. While the GPU will still be a bottleneck, looking back at the code, every action besides 'read YOLOv8 frame' uses the CPU. That means the CPU is not being utilised while the GPU is and vice versa. Multi-processing can solve this. Running multiple processes that are responsible for a video each will make it such that while a single process is blocked on the GPU the multiple other processes can use the CPU simultaneously. With these changes processing all the videos took ~2 days, roughly the real-time of the videos. Time to actually do something with the data.

## Training the predictors
While the data exploration and final outcomes are covered here ([ggstrive.ipynb](https://colab.research.google.com/drive/1ybJt9Y1jr8Qtdvq8T515--zxLptH8D7v?usp=sharing)) along with a greater explanation on the data format and eventual model I want to talk about the decisions and discoveries made along the way. There were two models that needed to be created the 'round predictor' and the 'set predictor'. Let's first look at the 'round predictor'.

### Feature Selection
During data collection many metrics were tracked that could be used however not all features are equal. It's usually a good idea to select the features will be most 'impactful' for predictions. What is considered impactful?

#### Data exploration
Doing some prelimnary graphing and comparing the different player's 'gauges', a clear trend is reveled.

![Data Exploration](../assets/img/data_exploration.png)
_Shockingly the player with more health is more likely going to win_

x-axis is the value of the gauge for P1, likewise y-axis is the value for gauge for P2. Blue indicates a P1 win and Red indicates a P2 win. So there is more blue when P1 approches 1.0 on the x-axis and vice versa. It's clear that health will be an important feature and will be selected however let's if there is any other trends we can gather.

![Data Exploration](../assets/img/data_exploration_burst.png)
![Data Exploration](../assets/img/data_exploration_tension.png)
![Data Exploration](../assets/img/data_exploration_risc.png)
_P1 vs P2 for various gauges_

The tension graph is seperated into 4 quardrants, this is due to how to implementation details of how tension was recorded visually however it becomes a useful side-effect when reviewing the graphs. Any action that uses *tension* requires the player to have at least 0.5 (50%), therefore in the graph the top half and right half represent when P2 and P1 can utilise their tension respectively. Top right and bottom left quardrants are both equal states, both players either can or can not use tension. However in the top left where only P2 can use tension there is more red dots, indicating more P2 wins however and vice versa for bottom right and P1 wins.

*Burst* and *R.I.S.C* graphs seem to have no obvious trend. On top of that R.I.S.C has the vast majority of datapoints around 0. This is unsurprising for those familiar with the game however it does put into question whether it is a useful feature.

For the gauges *health* is absolutely necessary, with *tension* also looking useful but *burst* and *R.I.S.C* needs to be investigated further

#### Statistical Tests
In doing some research I came across *[aziztitu's football match predictor](https://github.com/aziztitu/football-match-predictor)*. A project that trained a model to predict football matches with a helpful writeup attached. In it they detail the statistical tests that they perform including Chi^2, VIF for collinearity and variance.

Chi^2 Test
: From JMP
> The Chi-square test of independence is a statistical hypothesis test used to determine whether two categorical or nominal variables are likely to be related or not.[^footnote-chi]

The test is only useful for variables that are *frequencies*. It tries to see if a feature is dependent on the target variable by looking at the 'expected' frequency and the actual 'observered' frequency. The *gauges* were not evaluated with this however it could have been possible if the gauge values were 'binned', i.e. put into buckets/groups of values.

![chi^2 test](../assets/img/chi^2.png)
_Results of Chi^2 test analysis_

The **damaged** feature is quite significant as well as **counter** hit, however the rest don't look too impactful. This intuitively makes sense, the player that is damaged more is likely going to lose and being 'counter' hit usually results in serious damage. For **punish** as the data collection was performed on high-level play, players at that skill level don't put themselves in position to be punished as often. If it was performed on 'low-mid' level player this would likely be much more significant feature. 'Just' is rare in general requiring an instant block a technique requiring a 2-frame input, there just wasn't a lot of examples for it. **Reversal** seems to not matter this only appears if a player peforms a special move directly coming out of block stun or a knockdown but that's only useful in very specific situations.

Collinearity
: Using VIF (Variable Inflation Factor) each feature is checked for collinearity, that is how correlated two different features are. If two features are too similar, there is little reason to use both.
![chi^2 test](../assets/img/collinearity.png)
_VIF test output for features_

Although **p2_health** and **p1_burst** are in the 'removed features' section, it wouldn't make sense to remove a features only for a single player therefore these features would be kept. However it looks like **time** is highly collinear and will be removed.

Characters
: Character matchups are an essential part of any fighting game and should be essential to making an accurate prediction. Therefore **p1_name** and **p2_name** should be selected features. However after observing the results of some trained models the impact of the character choices seemed too signficant. The likely reason for this is simply there wasn't enough variety of matches observed. **Guilty Gear -Strive_** has 28 characters totalling in a possible 784 different matchups. There were around 238 videos used mostly only covering a single matchup each. Of the matchups that were represented there were likely only a few examples of them and between the same few players. Inadvertently it became more how the players matched up against each other rather than the characters. Due to that reasoning **p1_name** and **p2_name** were dropped.

The final feature list for the round predictor looks like:
| Column Name     | Type      |
| --------------- | --------- |
| p1_health       | `float`   |
| p2_health       | `float`   |
| p1_tension      | `float`   |
| p2_tension      | `float`   |
| p1_burst        | `float`   |
| p2_burst        | `float`   |
| p1_counter      | `int`     |
| p2_counter      | `int`     |
| p1_curr_damaged | `boolean` |
| p2_curr_damaged | `boolean` |

### Training the round predictor
The models Gaussian Naive Bayes, Logistic Regression, Random Forest Classifier, Decision Tree Classifier and Multi-Layered Perceptron were trained and compared by their accuracy. Initially the results looked very promising:

```
gb fit
Accuracy: 0.6347066037050760
lr fit
Accuracy: 0.6678068789587310
rfc fit
Accuracy: 0.9756247161339388
dtc fit
Accuracy: 0.7023490670238626
mlp fit
Accuracy: 0.6678068789587310
```

While the rest of the results were fine there was a clear winner, 97% accuracy with the Random Forest Classifier. This is beyond my wildest expectations for this project. Was it really possible to predict matches this accurately with only these few variables? Unfortunatley, if it sounds too good to be true, it probably is.

### Overfitting
A lesson for myself, be careful of code that you copy. When learning how to create the dataset I took a bunch of code from various sources. There was one line I glossed over though:

```python
#Split the data
round_x_train, round_x_test, round_y_train, round_y_test = train_test_split(
    round_x, round_y, test_size=0.33, random_state=125
)
```

Seemed innocent enough, it splits the data into training and testing data but `random_state=125` shuffled the data. The split should look like this:

![Training Test Split](../assets/img/training_test_split_intended.png)

However with the shuffle this happened:

![Training Test Split](../assets/img/shuffle.png)

Due to the shuffle, the random forest classifier was able to very accurately predict 'test' data as it was similar to the training data. For example, take these 3 data points:

|          | time  | p1_health | p2_health | p1_tension | p2_tension | p1_burst | p2_burst | p1_counter | p2_counter | p1_curr_damaged | p2 _curr_damaged | p1_round_win |
| -------- | ----- | --------- | --------- | ---------- | ---------- | -------- | -------- | ---------- | ---------- | --------------- | ---------------- | ------------ |
| Training | 10.2s | 78.7      | 65.9      | 50.0       | 22.0       | 100      | 11.9     | 2          | 1          | True            | False            | True         |
| Test     | 10.3s | 76.0      | 65.9      | 50.5       | 23.0       | 100      | 12.2     | 2          | 1          | True            | False            | True         |
| Training | 10.4s | 75.5      | 65.9      | 50.1       | 24.0       | 100      | 12.5     | 2          | 1          | True            | False            | True         |


The test data is situated in between the training data and it's values are very similar and have the same target, **player 1 wins**. The point for the model is to find the relationship between all the features that can predict the 'target'. It does this iteratively constantly adjusting values, verifying against the test data and then adjusting again. Since the test data is so similar to the training data, it becomes **overconfident** that these specific values across all features indicate a player 1 win i.e. when p1_health is 78.7 and p2_health is 65.9 and p1_tension is 50.0... indicates a player 1 win. This will cause the model to converge to match the outcomes of the games in the training set. It seems like Random Forest Classifiers was especially adept at this being ablt to build a multitide of decision trees to match these specific datapoints. This is a prime example of overfitting and it would be more correct to say that the model is recalling the matches in the dataset rather than trying to predict the outcome. While the 97% accuracy looks good, it isn't indidcative of how it will perform when viewing new matches which is the goal. Simply removing the `random_state=125` solves the overfitting. The random forest classifier drops to 65% accuracy, still a good result but much less exciting.

### Training the set predictor
This is mainly more of the same, I would highly recommend reading the notebook above if you're interested in more details however there is one notable thing to talk about. Determing the features was a little more challening. **Burst** and the **round count** were the only features that persist between rounds however these features alone would not be able to determine the winner of a set, it would need to know what is happening in the current round. It didn't seem right to use the same features as the **round predictor**, now with an added round count. Instead the prediction from the **round predictor** was fed into the **set predictor**.

The final features for the **set predictor** are:
*   `p1_burst`
*   `p2_burst`
*   `p1_round_count`
*   `p2_round_count`
*   `current_round_pred`
    * This is the result of the **round predictor** in the current game state

**Burst** is fed into both predictors but it's impact on a round and set are different.

After some hyper-parameter tuning the Multi-Layered Perceptron for both the round and set predictors was chosen moving forward. It's time to start looking at tournament matches...

## Asuka Arc
Although the results looked good, I wasn't fully satisfied by it. Especially after removing the character matchups from the features, the model seemed very generic only looking at a few UI elements. I wanted to see if there was something more character specific I could dive into and then I saw this:

{% include embed/youtube.html id='qYhKPmVlB0E' %}
_Credit: Sajam_

Asuka was an incredibly complicated character due to his spell mechanic. At any given time Asuka can store up to 4 spells to use. After using a spell he can draw more from a pool. Asuka's spells can be thought of as cards that he plays and draws from a deck. Using the techniques I've learnt so far would there be a way to determine the quality of the currently held spells? I can't believe I scope creeped myself...

### Synthetic Frames
Investigating Asuka's Spells requires another vision model but I refuse to draw more boxes. There is the additional issue that some spells are used much more often than others leading to a class distribution imbalance if I were to manually annotate game footage.


#### Footnotes
[^footnote]: https://en.wikipedia.org/wiki/Dunning%E2%80%93Kruger_effect
[^footnote-2]: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
[^footnote-3]: https://docs.opencv.org/3.4/db/d27/tutorial_py_table_of_contents_feature2d.html
[^footnote-4]: https://www.dustloop.com/
[^footnote-chi]: (https://www.jmp.com/en_au/statistics-knowledge-portal/chi-square-test/chi-square-test-of-independence.html)