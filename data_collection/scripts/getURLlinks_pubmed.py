from Bio import Entrez

def search_pubmed(query, max_results=10):
    Entrez.email = 'your.email@example.com'  # Always tell NCBI who you are
    handle = Entrez.esearch(db='pubmed', term=query, retmax=max_results)
    record = Entrez.read(handle)

    for id in record['IdList']:
        handle = Entrez.efetch(db='pubmed', id=id, rettype='xml', retmode='text')
        xml_text = handle.read().decode('utf-8')  # decode bytes to string
        if '<Article>' in xml_text:
            with open(f'{id}.txt', 'w') as f:  # save as .txt file
                f.write(xml_text)

# example usage
search_pubmed('nanotechnology')
