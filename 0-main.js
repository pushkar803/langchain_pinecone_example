// 1. Initialize a new project with: npm init -y, and create an 4 js files .env file 
// 2. npm i "@pinecone-database/pinecone@^0.0.10" dotenv@^16.0.3 langchain@^0.0.73
// 3. Obtain API key from OpenAI (https://platform.openai.com/account/api-keys)
// 4. Obtain API key from Pinecone (https://app.pinecone.io/)
// 5. Enter API keys in .env file
// Optional: if you want to use other file loaders (https://js.langchain.com/docs/modules/indexes/document_loaders/examples/file_loaders/)
import { PineconeClient } from "@pinecone-database/pinecone";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import * as dotenv from "dotenv";
import { createPineconeIndex } from "./1-createPineconeIndex.js";
import { updatePinecone } from "./2-updatePinecone.js";
import { queryPineconeVectorStoreAndQueryLLM } from "./3-queryPineconeAndQueryGPT.js";
import { PromptTemplate } from "langchain/prompts";

// 6. Load environment variables
dotenv.config();

// 7. Set up DirectoryLoader to load documents from the ./documents directory
const loader = new DirectoryLoader("./documents", {
    ".txt": (path) => new TextLoader(path),
    ".pdf": (path) => new PDFLoader(path),
});
const docs = await loader.load();

// 8. Set up variables for the filename, question, and index settings
const prompt  = PromptTemplate.fromTemplate(
  "rate all saying's from doc with most likely matching with `{input_text}` sort it by ascending order of rating and display names with ranking"
)
const query = "Im looking to learn French for my upcoming trip. I am a complete beginner with no knowledge of the language. Im looking for someone that can teach me with visaul aids. I would like to have 2 session per week, for the next month. I am available monday and friday evenings. My primary language is English. I would prefer a Male tutor";
const question = await  prompt.format({input_text:query})

const indexName = "pqr001";
const vectorDimension = 1536;

// 9. Initialize Pinecone client with API key and environment
const client = new PineconeClient();
await client.init({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_ENVIRONMENT,
});

// 10. Run the main async function
(async () => {

// 11. Check if Pinecone index exists and create if necessary
//  await createPineconeIndex(client, indexName, vectorDimension);

// 12. Update Pinecone vector store with document embeddings
//  await updatePinecone(client, indexName, docs);

// 13. Query Pinecone vector store and GPT model for an answer
  await queryPineconeVectorStoreAndQueryLLM(client, indexName, question);
})();
