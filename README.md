# Interview Assistant – RAG Chatbot with Azure Cognitive Search

This project is a Retrieval-Augmented Generation (RAG) chatbot designed to assist with interview preparation using personal documents such as resumes and job descriptions. It leverages Azure Cognitive Search and Azure OpenAI to deliver intelligent, contextual responses from custom data.

---

## Objective

Build a GenAI-powered assistant that can:
- Answer questions based on your resume, job description, company profile, interview tips, etc.
- Help you prepare for targeted interview questions
- Demonstrate LLM integration with Azure services

---

## What It Uses

- **Azure Cognitive Search** – For indexing and semantic retrieval
- **Azure OpenAI (or OpenAI API)** – For response generation via LLMs
- **Azure Cosmos DB** – For storing and retrieving embeddings
- **Azure Web App** – For hosting the Streamlit UI
- **Python** – For data preparation, chunking, and orchestration
- **Streamlit** – For the web-based user interface

---

## Deployment

### Required Environment Variables

In the Azure portal, go to **Settings > Environment variables**, and add the following environment variables:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDINGS_ENDPOINT`
- `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT`
- `COSMOS_DB_ENDPOINT`
- `COSMOS_DB_KEY`

To deploy in Azure (assuming you've set up OpenAI, embedding, and Cosmos resources):

1. Run `context.py` to generate embeddings and upsert them into Cosmos DB  
2. Zip the contents of the `code/` directory:  

```bash
cd code
zip -r ../code.zip .
```

3. Deploy with:  

```bash
az webapp deploy \
  --resource-group <YOUR-RESOURCE-GROUP> \
  --name <YOUR-WEB-APP> \
  --src-path ../code.zip \
  --type zip
```

4. In the Azure portal, go to your App Service > **Settings** > **Configuration**, and set this startup command:  

```bash
bash startup.sh
```
