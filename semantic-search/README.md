# rag-pipeline

A RAG Pipeline for retrival of information related to Food Ingredients provided query using the mixed-bread embedding model.
This project focuses on building a RAG setup to generate quality embeddings,  \
and to apply various techniques like multiprocessing and pyTorch DDP inorder  \
improve the time taken to generate these embeddings. 

The project focuses on:
- Scraping and parsing of large wikipedia articles, over 50K articles exclusively related to food ingredients.
- We employ python's multiprocessing module to speed up the parsing process and save the final parsed results as pickle dumps.
- Then we divide the articles in chuncks of size ~300 tokens on average based on context ( end of sentences or paragraphs )
- With the mixed bread (mxbai) embedding model, we embed these chunks to generate quality embeddings. 
- We used pytorch's DDP module to speed up the embedding process using 2 GPUs on NYU HPC.
- We save the embeddings, then index them with Faiss to use it as a vector database.
- Evaluating the quality of generated embeddings. 
- Finally, we integrate the retrival pipeline with gemma for fine tuning and inference. 

# Steps to generate the Embeddings

The following are the steps to recreate this project, to generate the embeddings from scratch.

## (1) Web scraping wikipedia articles and parse the scraped artcles related to Food Ingredients. 

```
python3 wiki_download.py
```

- We scrape over 500,000 wikipedia web links to look for specific articles with category : Food Ingredients.
- Download the articles and parse the raw text to a form that is suitable for our needs. 
- The raw articles are parsed as : { title : << title >> , site : << site link >> , text : << parsed text >> }
- We parsed over 67,000 articles, which takes more than 2 hours, on a single process.
- Therefore, we employ python's multiprocessing to parse these articles on 4 prcoesses which results in around 20 mins of runtime.
- The parsed articles are dumped as .pickle files, each processes dumps as a seperate pool, thus we end up with multiple dumps index from 0 to N - 1 processes.

## (2A) Split the articles into chunks and generate embeddings for these chunks. 

```
python3 mixed-bread-rag.py
```

- We then read in the dumps and split the articles into chunks that tries maintain sentence completions, thus we try to maintain some amount of context.
- In this case we chose to split the articles at an average of ~300 tokens per chunk. 
- Then we run the embedding model to generate embeddings for these chunks. The result is a dataset of 230k embeddings that takes over 2.5 hours to generate. 
- Finally, we index these embeddings using faiss for look up. ( 230k sections of articles that can be queried for look up by any llm model )

## (2B) Generate embeddings using pyTorch's DDP module to leverage multiple GPUs. 

```
python3 mixed-bread-rag-ddp.py
```

- We utilized pyTorch's DDP module to speed up the generation embeddings.
- Here we used 2 GPUs to generate the 230k embeddings in 1 hour. 
- The embedding datasets are saved as a pool by the individual processes. 

## (3) Finally test the quality of embeddings by using various examples.
``` 
python3 test-rag.py
```

- We supply queries and generate 10 nearest context articles using faiss indexing on our embedding dataset.
- The generated are as expected given the dataset and the embedding model. 

# Results:

### Query:

```
'sea foods that are good for bodybuilding'
```

### Retrieved related Wikipedia articles :

```
['Seafood', 'Dwaeji gukbap', 'Fufu', 'Lūʻau (food)', 'Atlantic surf clam', 'Pea protein', 'Fish sauce', 'Fish as food', 'Shrimp', 'Spiny dogfish']
```

### The sections of articles that are relavent to the given query :

```
Since 1960, annual global seafood consumption has more than doubled to over 20 kg per capita. Among the top consumers are Korea (78.5 kg per head), Norway (66.6 kg) and Portugal (61.5 kg).
The UK Food Standards Agency recommends that at least two portions of seafood should be consumed each week, one of which should be oil-rich. There are over 100 different types of seafood available around the coast of the UK.
Oil-rich fish such as mackerel or herring are rich in long chain Omega-3 oils. These oils are found in every cell of the human body, and are required for human biological functions such as brain functionality.
Whitefish such as haddock and cod are very low in fat and calories which, combined with oily fish rich in Omega-3 such as mackerel, sardines, fresh tuna, salmon and trout, can help to protect against coronary heart disease, as well as helping to develop strong bones and teeth.
Shellfish are particularly rich in zinc, which is essential for healthy skin and muscles as well as fertility. Casanova reputedly ate 50 oysters a day.

Texture and taste
Over 33,000 species of fish and many more marine invertebrate species have been identified. Bromophenols, which are produced by marine algae, give marine animals an odor and taste that is absent from freshwater fish and invertebrates. Also, a chemical substance called dimethylsulfoniopropionate (DMSP) that is found in red and green algae is transferred into animals in the marine food chain.
---------------------------------------------
 The dish may help health conditions due to its high nutrient content in things like calcium and protein.

---------------------------------------------
A dish called funche made with taro, green and yellow plantains boiled and mashed with butter, garlic, and pork fat was once popular in Puerto Rico. Once mashed it was formed into balls and eaten with broth made from sesame seeds. Funche is written in early Puerto Rican cookbooks around the 1800s, but can probably be traced back to African slaves on the island. Funche today in Puerto Rico is cornmeal cooked in coconut milk and milk.
The vegetable or fufú sauce in the Anglo-Caribbean is not fried first. Plantain is not used as much, as it is used in so many dishes. Fufu is usually part of, or added to, a soupy sauce or on the side with a soupy dish. In Antigua, fufu is served as part of the national dish but is called fungi/fungee and is made using cornmeal and okra. Similarly, in Barbados it serves as part of the national dish and is called cou-cou and uses cornmeal or, less commonly, split peas, green bananas, or breadfruit instead, like several other English Caribbean islands.

Nutrition
Nutritionally, 100 g dry weight fufu contains 2 g of protein, 0.1 g of fat and 84 g of carbohydrates. There are 267 kcal of food energy in a 100 g serving made up with water. It is low in cholesterol and rich in potassium, and it is commonly prescribed by doctors for people who have a low level of potassium in their blood.
---------------------------------------------
 "fatty")—rich foods that often contain a good amount of thicker coconut cream (not to be confused with sweetened "cream of coconut"). Beef, or povi ( lit. "bovine"), is the protein of choice in the form of brined povi masima (lit. "salted beef") or canned pīsupo (lit. "pea soup," general term for canned foods). Palusami is prepared by laying out a few taro leaves and spooning an amount of beef and onions into the center with a healthy amount of coconut cream and bundled with foil then steamed.

Tonga
Lū talo are typically prepared in parcels, in Tonga. Two popular versions are lū pulu (lit. "bull") refers to beef, and lū sipi (lit. "sheep") refers to mutton or lamb. Fresh meat can be used, corned (wet brine) masima or canned meats kapa are typical. Horse meat, hoosi, is also a delicacy. Coconut cream is often mixed into the meat, especially with canned meats, to form a paste that easily dollops. Chopped onions are common additions, sometimes tomatoes. Lū moa (chicken) and lū ika (fish) are made as well. The parcels are traditionally wrapped with banana leaves but it is more common to use foil. Kapisi pulu is a similar variant using kapisi (lit.
---------------------------------------------

About two-thirds of a surf clam's shucked weight is viable for human consumption. The meat of the clam is used as 'strips', chowder, and sushi.
The "tongue" or foot of the clam is commercially valuable because it is cut into long strips which are breaded and fried and served as clam strips, first popularized by the Howard Johnson's franchise.
The meat that is left over is separated from the "belly" and is referred to as "salvage" within the clam industry. This meat includes the adductor muscles, which are the strong muscles that close the two halves of the shell and which tightly hold the clam's shell in the shut position. "Salvage" is typically ground up for use in chowders, sauces, and dips, and is commercially available either in cans or frozen. Locally it is available fresh. The substantial "belly" of the clam is used by some fishermen as bait for striped bass and other species.

---------------------------------------------
See also
Pea milk
Soy protein
Bodybuilding supplement
Whey protein

---------------------------------------------
 Ikanago shoyu of Kagawa Prefecture is made from sand lance. They are used in nabemono, in salad dressings, and as a flavoring ingredient in ramen soups.

Korea
In Korea, fish sauce is called eojang (어장).
Across the Korean Peninsula, aekjeot (액젓, literally "liquid jeotgal"), a type of fish sauce usually made from fermented anchovies or sand lances, is used as a crucial ingredient in many types of kimchi, both for taste and fermentation.
In Jeju island, eoganjang (어간장), made of fermented godori (young chub mackerel) or horse mackerel, is used in place of soy sauce.

Europe
Italy
Colatura di alici is an Italian fish sauce originating in the village of Cetara, Campania.

England
Worcestershire sauce contains fermented anchovies among other ingredients, which is common in the Anglosphere countries.

Nutrition contents
Common commercial brands of fish sauce generally contain about 50% to 60% of the FDA's daily recommended amount of sodium per tablespoon serving. Most commercial brands of reasonable quality contain one or two grams of protein per serving; however, higher-quality brands may have four grams of protein or more, while lower-quality brands may have less than one gram of protein per serving. Fish sauce has an insignificant amount of carbohydrates and fats.
---------------------------------------------

The British historian William Radcliffe wrote in Fishing from the Earliest Times:

"The Emperor Domitian (Juvenal, IV.) ordered a special sitting of the Senate to deliberate and advise on a matter of such grave State importance as the best method of cooking a turbot."

Nutritional value
Globally, fish and fish products provide an average of only about 34 calories per capita per day. However, more than as an energy source, the dietary contribution of fish is significant in terms of high-quality, easily digested animal proteins and especially in fighting micronutrient deficiencies. A portion of 150g of fish provides about 50 to 60 percent of an adult's daily protein requirement. Fish proteins are essential in the diet of some densely populated countries where the total protein intake is low, and are particularly important in diets in small island developing States (SIDS).
Intermediate Technology Publications wrote in 1992 that "Fish provides a good source of high quality protein and contains many vitamins and minerals. It may be classed as either whitefish, oily fish, or shellfish. Whitefish, such as haddock and seer, contain very little fat (usually less than 1%) whereas oily fish, such as sardines, contain between 10–25%. The latter, as a result of its high fat content, contain a range of fat-soluble vitamins (A, D, E and K) and essential fatty acids, all of which are vital for the healthy functioning of the body.
---------------------------------------------
 Usually shrimp is sold whole, though sometimes only the meat of shrimp is marketed.
As with other seafood, shrimp is high in calcium, iodine and protein but low in food energy. A shrimp-based meal is also a significant source of cholesterol, from 122 mg to 251 mg per 100 g of shrimp, depending on the method of preparation. Shrimp consumption, however, is considered healthy for the circulatory system because the lack of significant levels of saturated fat in shrimp means that the high cholesterol content in shrimp improves the ratio of LDL to HDL cholesterol and lowers triglycerides.
Ebiko - shrimp roe, sometimes translated as "shrimp flakes" - is used as one of the ingredients in the preparation of sushi.
Shrimp and other shellfish are among the most common food allergens. They are not kosher and thus are forbidden in Jewish cuisine.

Aquaria
Several types of shrimp are kept in home aquaria. Some are purely ornamental, while others are useful in controlling algae and removing debris. Freshwater shrimp commonly available for aquaria include the Bamboo shrimp, Japanese marsh shrimp (Caridina multidentata, also called "Amano shrimp," as their use in aquaria was pioneered by Takashi Amano), cherry shrimp (Neocaridina heteropoda), and ghost or glass shrimp (Palaemonetes spp.). Popular saltwater shrimp include the cleaner shrimp Lysmata amboinensis, the fire shrimp (Lysmata debelius) and the harlequin shrimp (Hymenocera picta).
---------------------------------------------

Life span estimates based on analysis of vertebral centra and annuli in the dorsal spines range from 35 to 54 years.

Commercial use
Spiny dogfish are sold as food in Europe, the United States, Canada, New Zealand, Venezuela and Chile. The meat is primarily eaten in England, France, Italy, the Benelux countries, Germany, and as an unwitting houndshark in Greece. The fins and tails are processed into fin needles for cheaper versions of shark fin soup in Chinese cuisine. In England, dogfish are bought in fish and chip shops as "huss", and previously as "rock salmon" until the term was outlawed. Commercial harvests were banned in the UK from 2010 until 2023. In France, they are sold as "small salmon" (saumonette) and in Belgium and Germany as "sea eel" (zeepaling and Seeaal, respectively). In Greece, all small dogfish are commercially called galéos, after the critically endangered Galeorhinus galeus that produces a seasonal dish; other small sharks such as the spiny dogfish substitute it. 
Spiny dogfish bodies are ground into fertilizer, liver oil and pet food. Because of their availability, cartilaginous skulls and small sizes, they are popular vertebrate dissection specimens in high schools and universities.
Reported catches varied between 31,700 tonnes in 2000 and 13,800 tonnes in 2008. Bottom trawlers and sink gillnets are the primary tools.
---------------------------------------------
```

# Conclusions:

Our RAG pipeline works fine.
We were able to leverage multiprocessing and DDP to work with larger datasets and generate even larger amount of embeddings.
