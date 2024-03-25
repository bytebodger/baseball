import { dbClient } from '../../constants/dbClient.js';

export const getDBOldestUnprocessedBoxscore = async () => {
   return await dbClient.query(`
      SELECT
         web_boxscore.html
         ,web_boxscore.season
         ,web_boxscore.url
         ,web_boxscore.web_boxscore_id
      FROM
         web_boxscore
      WHERE
         web_boxscore.html IS NOT NULL
      AND 
         web_boxscore.time_retrieved IS NOT NULL
      AND
         web_boxscore.time_processed IS NULL
      ORDER BY
         web_boxscore.web_boxscore_id ASC
      LIMIT
         1
   `)
}