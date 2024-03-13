import { parse } from 'node-html-parser';
import { WebBoxscore } from '../interfaces/tables/WebBoxscore.js';
import { getString } from './getString.js';
import { getOldestUnprocessedBoxscore } from './queries/getOldestUnprocessedBoxscore.js';

export const processBoxScores = () => {
   (async () => {
      const { rows: boxscores } = await getOldestUnprocessedBoxscore() as { rows: WebBoxscore[] };
      if (!boxscores.length)
         return;
      const { html, season, url } = boxscores[0];
      if (!html)
         return;
      const baseballReferenceId = getString(url.split('/').pop()).split('.').shift();
      const dom = parse(html);

   })()
}