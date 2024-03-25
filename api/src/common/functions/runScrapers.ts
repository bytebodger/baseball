import { pageDelay } from '../constants/pageDelay.js';
import { runProcess } from './runProcess.js';
import { wait } from './wait.js';

export const runScrapers = async () => {
   let proceed = true;
   while (proceed) {
      const result = await runProcess('dist\\common\\functions\\run\\runScrapeWebSchedule.js');
      console.log(result);
      proceed = result.proceed;
      if (proceed)
         await wait(pageDelay);
   }
   proceed = true;
   while (proceed) {
      const result = await runProcess('dist\\common\\functions\\run\\runScrapeWebBoxscore.js');
      console.log(result);
      proceed = result.proceed;
      if (proceed)
         await wait(pageDelay);
   }
}