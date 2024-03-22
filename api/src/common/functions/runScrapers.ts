import { pageDelay } from '../constants/pageDelay.js';
import { runProcess } from './runProcess.js';
import { wait } from './wait.js';

export const runScrapers = async () => {
   let proceed = true;
   while (proceed) {
      const result = await runProcess('dist\\common\\functions\\run\\runRetrieveWebSchedules.js');
      proceed = result.proceed;
      if (proceed)
         await wait(pageDelay);
   }
}