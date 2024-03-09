import cors from 'cors';
import express, { Express, Request, Response } from 'express';
import { Endpoint } from './common/enums/Endpoint.js';
import { scrape } from './common/functions/scrape.js';

const app: Express = express();
const port = process.env.PORT;
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
scrape();

app.get(
   Endpoint.root,
   (_request: Request, response: Response) => response.status(200).send('Baseball AI API'),
)

app.listen(port, () => {
   console.log(`⚡️[server]: Server is running at http://localhost:${port}`);
});