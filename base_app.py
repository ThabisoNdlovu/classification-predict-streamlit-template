"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from streamlit_option_menu import option_menu
from wordcloud import WordCloud
from prettytable import PrettyTable
import matplotlib.pyplot as plt

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# Displaying the image
image = "resources/logo.jpg"  # Replace with the path to your image file
st.image(image, width=200)

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Climate Conscious Consulting")




	

	st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
	)
	st.markdown('<p class="centered-title">From data chaos to strategic insights: Empowering businesses through analytics</p>', unsafe_allow_html=True)


	
#	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	with st.sidebar:
		selection = option_menu("Main Menu", ["Home", 'Visualisation', 'The Team','Contact Us','About'], 
        icons=['house', 'pie-chart', 'people-fill', 'envelope','info-circle'], menu_icon="cast", default_index=1)
	#options = ["Prediction", "Information", "Development Team","Information 2" ]
	#selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page


	# Building out the predication page
	if selection == "Home":
		st.subheader("Let's classify")
	

		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			prediction_dic =  {-1:"Anti: the tweet does not believe in man-made climate change", 0:"Neutral: the tweet neither supports nor refutes the belief of man-made climate change",
			1:"Pro: the tweet supports the belief of man-made climate change", 2:"News: the tweet links to factual news about climate change"}
			st.success("Text Categorized as: {}".format(prediction_dic[prediction[0]]))
	if selection == "The Team":
		st.title("")
		job_titles = [
		("Ntsika Xuba", "CEO","aaaaaaaaaaaaaaaa"),
		("Precious Sefike", "Data Scientist","bbbbbbbbbbbbbbbbb"),
		("Seshwene Makhura" , "Machine Learning Engineer","ccccccccccccccccc"),
		("Zama Dlamini" ,"Data Analyst","ddddddddddddddd"),
		("Mahlori Nkuna" , "Data Scientist","eeeeeeeeeeeeeeeeee"),
		("Thabiso Ndlovu" , "App Developer","fffffffffffffffff")]

		table = PrettyTable()
		table.field_names = ["", "Position", "LinkedIn"]

		for name, position,linkedin in job_titles:
    			table.add_row([name, position,linkedin])

		st.write(table)




	if selection == "Visualisation":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			
			st.write(raw[['sentiment', 'message']].head()) # will write the df to the page

			opt = st.radio('Plot  type:',['Bar', 'Pie', 'Word Cloud'])
			if opt=='Bar':
				st.markdown('<h3>Show sentiment occurance dataset</h3>',unsafe_allow_html=True)
				xx = raw['sentiment'].value_counts()
				st.bar_chart(xx)
			elif opt =="Pie":
				st.markdown('<h3>Pie chart for percentage of each sentiment on dataset</h3>',unsafe_allow_html=True)
				fig1, ax1 = plt.subplots()
				ax1.pie(raw['sentiment'].value_counts(),labels = ["Pro","News","Neutral","Anti"], autopct='%1.1f%%',shadow=True, startangle=90)
				ax1.axis('equal')
				ax1.set_facecolor("black")  # Equal aspect ratio ensures that pie is drawn as a circle.
				ax1.legend()
				fig1.patch.set_alpha(0)
				ax1.xaxis.label.set_color('red')
				st.pyplot(fig1)
				
		
			else:
				st.set_option('deprecation.showPyplotGlobalUse', False)
				st.markdown('<h3>Word Cloud for how frequently words show up on all tweets.</h3>',unsafe_allow_html=True)
				allwords = ' '.join([msg for msg in raw['message']])
				WordCloudtest = WordCloud(width = 800, height=500, random_state = 21 , max_font_size =119).generate(allwords)
				
				plt.imshow(WordCloudtest, interpolation = 'bilinear')
				
				plt.axis('off')
				st.pyplot(plt.show())
	if selection =="Contact Us":
		col1, mid, col2 = st.columns([80,1,80])
		st.write("Email: ")
		st.write("Phone: ")
		st.write("Address: ")

	if selection=="About":
			st.write("At Climate Conscious Consulting, we are dedicated to empowering businesses and organizations to make meaningful strides towards a sustainable future. We understand that climate change is one of the most pressing challenges of our time.") 
			st.write("Our mission is to guide and support companies in navigating the complexities of sustainability, helping them integrate environmentally conscious practices into their operations and decision-making processes. We recognize that addressing climate change requires a holistic approach, encompassing not only environmental considerations but also social and economic factors. By considering the triple bottom line—people, planet, and profit—we help our clients create long-term value while minimizing their environmental impact.")
			st.write("With our expert team of sustainability consultants, we offer a comprehensive range of services tailored to meet the unique needs of each organization. From conducting thorough environmental assessments to developing and implementing sustainable strategies, we provide the tools and knowledge necessary to drive positive change. Our solutions are designed to optimize resource efficiency, reduce greenhouse gas emissions, and enhance overall sustainability performance.")
			st.write("We are committed to staying at the forefront of sustainability trends and best practices. We continuously research and analyze the latest advancements in the field using cutting edge technologies, ensuring that our clients receive cutting-edge insights and innovative solutions. We believe in collaboration and actively engage with our clients, working hand in hand to develop strategies that align with their goals and values.")  
			st.write("By choosing Climate Conscious Consulting as your sustainability partner, you can be confident that you are working with a team that is passionate about creating a more sustainable future. Together, we can make a significant impact and foster a world where businesses thrive while respecting the planet and its inhabitants.")      
			st.write("Join us on this journey towards a climate-conscious future. Contact us today to learn more about how we can help your organization embrace sustainability and be a force for positive change.")


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
