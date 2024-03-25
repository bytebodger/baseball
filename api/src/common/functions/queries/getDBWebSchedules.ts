import { dbClient } from '../../constants/dbClient.js';

export const getDBWebSchedules = async () => {
   return await dbClient.query(`
      SELECT
         web_schedule.has_been_played
         ,web_schedule.html
         ,web_schedule.season
         ,web_schedule.time_processed
         ,web_schedule.time_retrieved
         ,web_schedule.url
         ,web_schedule.web_schedule_id
      FROM 
         web_schedule
      ORDER BY
         web_schedule.season DESC
   `)
}