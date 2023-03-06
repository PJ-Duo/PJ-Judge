module Spacy
#https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/
using Conda
using PyCall

export spacy
const python3 = joinpath(Conda.python_dir(Conda.ROOTENV), "python3")
const spacy = pyimport("spacy")
const spacyTokens = pyimport("spacy.tokens")

function blank(model::String="en_core_web_sm")
  spacy.blank(model)
end

function DocBin()
  return spacyTokens.DocBin()
end

end

function matcher(nlp)
    spacyMatcher.matcher(nlp)
end

nlp = Spacy.load("en_core_web_md")
db = Spacy.DocBin()

nlp.pipe_names

article_text="""India that previously comprised only a handful of players in the e-commerce space, is now home to many biggies and giants battling out with each other to reach the top. This is thanks to the overwhelming internet and smartphone penetration coupled with the ever-increasing digital adoption across the country. These new-age innovations not only gave emerging startups a unique platform to deliver seamless shopping experiences but also provided brick and mortar stores with a level-playing field to begin their online journeys without leaving their offline legacies.
In the wake of so many players coming together on one platform, the Indian e-commerce market is envisioned to reach USD 84 billion in 2021 from USD 24 billion in 2017. Further, with the rate at which internet penetration is increasing, we can expect more and more international retailers coming to India in addition to a large pool of new startups. This, in turn, will provide a major Philip to the organized retail market and boost its share from 12% in 2017 to 22-25% by 2021. 
Here's a view to the e-commerce giants that are dominating India's online shopping space:
Amazon - One of the uncontested global leaders, Amazon started its journey as a simple online bookstore that gradually expanded its reach to provide a large suite of diversified products including media, furniture, food, and electronics, among others. And now with the launch of Amazon Prime and Amazon Music Limited, it has taken customer experience to a godly level, which will remain undefeatable for a very long time. 

Flipkart - Founded in 2007, Flipkart is recognized as the national leader in the Indian e-commerce market. Just like Amazon, it started operating by selling books and then entered other categories such as electronics, fashion, and lifestyle, mobile phones, etc. And now that it has been acquired by Walmart, one of the largest leading platforms of e-commerce in the US, it has also raised its bar of customer offerings in all aspects and giving huge competition to Amazon. 

Snapdeal - Started as a daily deals platform in 2010, Snapdeal became a full-fledged online marketplace in 2011 comprising more than 3 lac sellers across India. The platform offers over 30 million products across 800+ diverse categories from over 125,000 regional, national, and international brands and retailers. The Indian e-commerce firm follows a robust strategy to stay at the forefront of innovation and deliver seamless customer offerings to its wide customer base. It has shown great potential for recovery in recent years despite losing Freecharge and Unicommerce. 

ShopClues - Another renowned name in the Indian e-commerce industry, ShopClues was founded in July 2011. It`s a Gurugram based company having a current valuation of INR 1.1 billion and is backed by prominent names including Nexus Venture Partners, Tiger Global, and Helion Ventures as its major investors. Presently, the platform comprises more than 5 lac sellers selling products in nine different categories such as computers, cameras, mobiles, etc. 

Paytm Mall - To compete with the existing e-commerce giants, Paytm, an online payment system has also launched its online marketplace - Paytm Mall, which offers a wide array of products ranging from men and women fashion to groceries and cosmetics, electronics and home products, and many more. The unique thing about this platform is that it serves as a medium for third parties to sell their products directly through the widely-known app – Paytm. 

Reliance Retail - Given Reliance Jio's disruptive venture in the Indian telecom space along with a solid market presence of Reliance, it is no wonder that Reliance will soon be foraying into retail space. As of now, it has plans to build an e-commerce space that will be established on online-to-offline market program and aim to bring local merchants on board to help them boost their sales and compete with the existing industry leaders. 
Big Basket - India's biggest online supermarket, Big Basket provides a wide variety of imported and gourmet products through two types of delivery services - express delivery and slotted delivery. It also offers pre-cut fruits along with a long list of beverages including fresh juices, cold drinks, hot teas, etc. Moreover, it not only provides farm-fresh products but also ensures that the farmer gets better prices. 

Grofers - One of the leading e-commerce players in the grocery segment, Grofers started its operations in 2013 and has reached overwhelming heights in the last 5 years. Its wide range of products includes atta, milk, oil, daily need products, vegetables, dairy products, juices, beverages, among others. With its growing reach across India, it has become one of the favorite supermarkets for Indian consumers who want to shop grocery items from the comforts of their homes. 

Digital Mall of Asia - Going live in 2020, Digital Mall of Asia is a very unique concept coined by the founders of Yokeasia Malls. It is designed to provide an immersive digital space equipped with multiple visual and sensory elements to sellers and shoppers. It will also give retailers exclusive rights to sell a particular product category or brand in their respective cities. What makes it unique is its zero-commission model enabling retailers to pay only a fixed amount of monthly rental instead of paying commissions. With its one-of-a-kind features, DMA is expected to bring
never-seen transformation to the current e-commerce ecosystem while addressing all the existing e-commerce worries such as counterfeiting. """

doc = nlp(article_text)

print()

for (ent) in doc.ents
    println("jio")
end