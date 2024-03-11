import { dbClient } from '../../constants/dbClient.js';

export const getWebSchedules = async () => {
   return await dbClient.query(`
      SELECT
         web_schedule.html
         ,web_schedule.is_complete
         ,web_schedule.season
         ,web_schedule.time_retrieved
         ,web_schedule.url
         ,web_schedule.web_schedule_id
      FROM 
         web_schedule
      ORDER BY
         web_schedule.season DESC
   `)
}