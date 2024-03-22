import { retrieveWebSchedules } from './retrieveWebSchedules.js';

export const scrape = async () => {

   await retrieveWebSchedules();
   //await retrieveWebBoxscores();
   //await processBoxScores();
}