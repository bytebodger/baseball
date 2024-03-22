import cors from 'cors';
import dayjs from 'dayjs';
import dayOfYear from 'dayjs/plugin/dayOfYear.js';
import isSameOrAfter from 'dayjs/plugin/isSameOrAfter.js';
import isSameOrBefore from 'dayjs/plugin/isSameOrBefore.js';
import utc from 'dayjs/plugin/utc.js';
import type { Express } from 'express';
import express from 'express';
import { dbClient } from '../constants/dbClient.js';

export const initialize = async () => {
   await dbClient.connect();
   const api: Express = express();
   const port = process.env.PORT;
   api.use(cors());
   api.use(express.json());
   api.use(express.urlencoded({ extended: true }));
   dayjs.extend(dayOfYear);
   dayjs.extend(utc);
   dayjs.extend(isSameOrAfter);
   dayjs.extend(isSameOrBefore);
   api.listen(port, () => console.log(`⚡️[server]: Server is running at http://localhost:${port}`));
   return api;
}