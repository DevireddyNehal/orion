from ddgs import DDGS

with DDGS(timeout=100) as ddgs:
    news_results = list(ddgs.text("Weather in Horamavu Agara", max_results=3))
    
    print(news_results)
        
