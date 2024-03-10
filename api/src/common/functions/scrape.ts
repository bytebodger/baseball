import dayjs from 'dayjs';
import type { WebSchedule } from '../interfaces/tables/WebSchedule.js';
import { getLatestWebSchedule } from './queries/getLatestWebSchedule.js';

export const scrape = () => {
   (async () => {
      const currentYear = dayjs().utc().year();
      const { rowCount, rows: webSchedule } = await getLatestWebSchedule() as { rowCount: number, rows: WebSchedule[] };
      const nextSeason = rowCount ? webSchedule[0].season + 1 : process.env.FIRST_YEAR;
   })()
}