import express from 'express';
import cors from 'cors';
import { Endpoint } from './common/enums/Endpoint.js';

const app = express();
const port = process.env.PORT;
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.get(Endpoint.root, (_request, response) => response.status(200).send('Baseball AI API'));
app.listen(port, () => {
   console.log(`⚡️[server]: Server is running at http://localhost:${port}`);
});
