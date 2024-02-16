import streamlit as st

def chat():
    st.markdown("<h2 style='text-align: center; color: Black;'>Chat With Our Off The Shelf Code Assistant</h2>", unsafe_allow_html=True)
    _ , chat_col, _ = st.columns([4, 5, 1])
    with chat_col:
      st.markdown('[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customizable-code-assistant-chat.streamlit.app/)', unsafe_allow_html=True)
    st.info("Please note to limit costs, the chatbot goes to sleep after 15 minutes of inactivity, this may delay the first response")
    st.markdown('---')
    st.markdown("<h2 style='text-align: center; color: Black;'>How to use the API</h2>", unsafe_allow_html=True)
    api_help_models = {
      "Security Model":{

        "python": '''
import requests

API_URL = "https://oa6kdk8gxzmfy79k.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Accept" : "application/json",
	"Content-Type": "application/json" 
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

output = query({
	"inputs": "Can you please let us know more details about your ",
	"config": {}
})
        ''',
        "javascript": '''
        async function query(data) {
	const response = await fetch(
		"https://oa6kdk8gxzmfy79k.us-east-1.aws.endpoints.huggingface.cloud",
		{
			headers: { 
				"Accept" : "application/json",
				"Content-Type": "application/json" 
			},
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.json();
	return result;
}

query({
    "inputs": "Can you please let us know more details about your ",
    "config": {}
}).then((response) => {
	console.log(JSON.stringify(response));
});
        ''',
        "curl": '''
        curl "https://oa6kdk8gxzmfy79k.us-east-1.aws.endpoints.huggingface.cloud" \
-X POST \
-d '{
    "inputs": "Can you please let us know more details about your ",
    "config": {}
}' \
-H "Accept: application/json" \
-H "Content-Type: application/json"
        '''
      },
      "React Model": {
        "python": '''
        import requests

API_URL = "https://s6izgwncr4ncciig.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Accept" : "application/json",
	"Authorization": "Bearer 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",
	"Content-Type": "application/json" 
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

output = query({
	"inputs": "Can you please let us know more details about your ",
	"parameters": {}
})
        ''',
        "javascript": '''
        async function query(data) {
	const response = await fetch(
		"https://s6izgwncr4ncciig.us-east-1.aws.endpoints.huggingface.cloud",
		{
			headers: { 
				"Accept" : "application/json",
				"Authorization": "Bearer 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",
				"Content-Type": "application/json" 
			},
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.json();
	return result;
}

query({
    "inputs": "Can you please let us know more details about your ",
    "parameters": {}
}).then((response) => {
	console.log(JSON.stringify(response));
});
        ''',
        "curl": '''
        curl "https://s6izgwncr4ncciig.us-east-1.aws.endpoints.huggingface.cloud" \
-X POST \
-H "Accept: application/json" \
-H "Authorization: Bearer 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" \
-H "Content-Type: application/json" \
-d '{
    "inputs": "Can you please let us know more details about your ",
    "parameters": {}
}'
        '''
      } 
    }
    #let the user choose the model
    models_options = list(api_help_models.keys())
    model = st.selectbox("Choose the model you want to use", models_options)
    if model == "Security Model":
      st.info("The security model is free to use but requires a Token from Hugging Face")
    lang_col, help_col = st.columns([1, 5])

    api_help = api_help_models[model]
    help_text, help_lang = api_help["python"], "python"
    with lang_col:
      python_btn = st.button("Python", key="python", type="primary" )
      js_btn = st.button("Javascript", key="javascript", type="primary")
      curl_btn = st.button("Curl", key="curl", type="primary")
    if python_btn:
      help_text, help_lang = api_help["python"], "python"
    if js_btn:
      help_text, help_lang = api_help["javascript"], "javascript"
    if curl_btn:
      help_text, help_lang = api_help["curl"], "curl"
    with help_col:
      st.code(help_text, language=help_lang)

    st.markdown('---')
    st.markdown("<h2 style='text-align: center; color: Black;'>Train Your Own Model With Our Python Package</h2>", unsafe_allow_html=True)
    #add link to the python package and colab notebook demo
    _ , col_pip,_, col_colab, _ = st.columns([1, 2, 2, 2, 1])
    with col_pip:
      st.markdown('[![PyPI version](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://pypi.org/project/enigma-ai/0.2.1/)', unsafe_allow_html=True)
    with col_colab:
      st.markdown('[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ccE5vUMitMBBBAKvXaKLMqN9zB1Sc2Ri?usp=sharing)', unsafe_allow_html=True)
    