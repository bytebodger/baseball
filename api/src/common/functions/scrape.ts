import { getLatestWebSchedule } from './queries/getLatestWebSchedule.js';

export const scrape = () => {
   (async () => {
      const latestWebSchedule = await getLatestWebSchedule();
      console.log('latestWebSchedule', latestWebSchedule);
   })()
}