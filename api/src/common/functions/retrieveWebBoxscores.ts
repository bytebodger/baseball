import dayjs from 'dayjs';
import { Milliseconds } from '../enums/Milliseconds.js';
import { WebBoxscore } from '../interfaces/tables/WebBoxscore.js';
import { getHttps } from './getHttps.js';
import { getOldestUnprocessedBoxscore } from './queries/getOldestUnprocessedBoxscore.js';
import { updateWebBoxscore } from './queries/updateWebBoxscore.js';

export const retrieveWebBoxscores = async () => {
   const { rows: boxscores } = await getOldestUnprocessedBoxscore() as { rows: WebBoxscore[] };
   if (!boxscores.length)
      return;
   const html = await getHttps(boxscores[0].url);
   await updateWebBoxscore({
      html,
      time_retrieved: dayjs().utc().valueOf(),
      web_boxscore_id: boxscores[0].web_boxscore_id,
   })
   setTimeout(() => retrieveWebBoxscores(), 4 * Milliseconds.second);
}