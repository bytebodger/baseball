import dayjs from 'dayjs';
import * as https from 'https';
import { Milliseconds } from '../enums/Milliseconds.js';
import type { WebBoxscore } from '../interfaces/tables/WebBoxscore.js';
import { getOldestUnprocessedBoxscore } from './queries/getOldestUnprocessedBoxscore.js';
import { updateWebBoxscore } from './queries/updateWebBoxscore.js';

export const retrieveWebBoxscores = () => {
   (async () => {
      const { rows: boxscores } = await getOldestUnprocessedBoxscore() as { rows: WebBoxscore[] };
      if (!boxscores.length)
         return;
      const { url, web_boxscore_id } = boxscores[0];
      https.get(url, response => {
         let html = '';
         response.on('data', chunk => {
            html += chunk;
         });
         response.on('end', () => {
            (async () => {
               await updateWebBoxscore({
                  html,
                  time_retrieved: dayjs().utc().valueOf(),
                  web_boxscore_id,
               })
               setTimeout(() => retrieveWebBoxscores(), 4 * Milliseconds.second);
            })()
         })
      })
   })()
}