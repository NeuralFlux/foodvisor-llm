import multiprocessing
from multiprocessing import Pool
from urllib.parse import quote
from tqdm import tqdm
import wikipediaapi
from collections import deque
import pickle
import os
import shutil

dump_file = "wiki_dump"
dir = "dumps_1"

def _construct_url(title, language):
    # See: https://meta.wikimedia.org/wiki/Help:URL
    return f"https://{language}.wikipedia.org/wiki/{quote(title)}"

def _get_dict_from_pages(process):

    f = open(f"{dir}/{dump_file}_{process}.pickle", 'rb')
    pages = pickle.load(f)
    f.close()

    wiki_data = []
    max_iters = len(pages)
    for i in tqdm(range(max_iters)):
        wiki_data.append({
            'title': pages[i].title,
            'site': _construct_url(pages[i].title, 'en'),
            'text': pages[i].text
        })
    
    with open(f'{dir}/{dump_file}_out_{process}.pickle', 'wb') as f:
        pickle.dump(wiki_data, f)
    


def scrape_category(category, max_pages=50000, pbar=None):
    categories = deque()
    categories.append(category)

    pages = deque(maxlen=max_pages)
    while len(categories) != 0 and len(pages) < max_pages:
        category = categories.popleft()

        for page in category.categorymembers.values():
            if page.ns == wikipediaapi.Namespace.CATEGORY:
                categories.append(page)
            else:
                pages.append(page)

                if pbar:
                    pbar.update(1)

    return pages


if __name__ == '__main__':

    wiki_wiki = wikipediaapi.Wikipedia('MerlinColab (merlin@example.com)', 'en')
    cat = wiki_wiki.page("Category:Food ingredients")
    
    NUM_SCRAPE_PAGES = 500_000 
    with tqdm(total=NUM_SCRAPE_PAGES) as pbar:
       pages = scrape_category(cat, NUM_SCRAPE_PAGES, pbar)
    
    pool = Pool()

    pages = list(pages)
    pages_set = list(set([pages[i].title for i in range(NUM_SCRAPE_PAGES)]))
    processes = 4
    print("Total number of pages : ", len(pages_set))

    pages_list = []
    for page in pages:
        if page.title in pages_set:
            pages_list.append(page)
            pages_set.remove(page.title)

    pages = pages_list
    print("Pages after removing duplicates : ", len(pages))
   
    div = int(len(pages)/processes) 
    args = [pages[x * div: (x * div) + div] for x in range(0, processes)]
   
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    
    for i in range(processes):
        with open(f'{dir}/{dump_file}_{i}.pickle', 'wb') as f:
            pickle.dump(args[i], f)

    pool.map(_get_dict_from_pages, [i for i in range(processes)])
    
    print(f"\n done ..   \n")
    
    
     
